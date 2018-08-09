# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import mock
import elasticsearch_dsl
import pytest
from hypothesis import strategies as st
from hypothesis import given
from webob import multidict

from h.search import Search, query

ES_VERSION = (1, 7, 0)
MISSING = object()

OFFSET_DEFAULT = 0
LIMIT_DEFAULT = 20
LIMIT_MAX = 200


class IndividualQualifiers(object):
    @pytest.mark.parametrize('offset,from_', [
        # defaults to OFFSET_DEFAULT
        (MISSING, OFFSET_DEFAULT),
        # straightforward pass-through
        (7, 7),
        (42, 42),
        # string values should be converted
        ("23", 23),
        ("82", 82),
        # invalid values should be ignored and the default should be returned
        ("foo",  OFFSET_DEFAULT),
        ("",     OFFSET_DEFAULT),
        ("   ",  OFFSET_DEFAULT),
        ("-23",  OFFSET_DEFAULT),
        ("32.7", OFFSET_DEFAULT),
    ])
    def test_offset(self, offset, from_, search):
        if offset is MISSING:
            params = {}
        else:
            params = {"offset": offset}

        search = query.Limiter()(search, params)
        q = search.to_dict()

        assert q["from"] == from_

    @given(st.text())
    @pytest.mark.fuzz
    def test_limit_output_within_bounds(self, text, search):
        """Given any string input, output should be in the allowed range."""
        params = {"limit": text}

        search = query.Limiter()(search, params)
        q = search.to_dict()

        assert isinstance(q["size"], int)
        assert 0 <= q["size"] <= LIMIT_MAX

    @given(st.integers())
    @pytest.mark.fuzz
    def test_limit_output_within_bounds_int_input(self, lim, search):
        """Given any integer input, output should be in the allowed range."""
        params = {"limit": str(lim)}

        search = query.Limiter()(search, params)
        q = search.to_dict()

        assert isinstance(q["size"], int)
        assert 0 <= q["size"] <= LIMIT_MAX

    @given(st.integers(min_value=0, max_value=LIMIT_MAX))
    @pytest.mark.fuzz
    def test_limit_matches_input(self, lim, search):
        """Given an integer in the allowed range, it should be passed through."""
        params = {"limit": str(lim)}

        search = query.Limiter()(search, params)
        q = search.to_dict()

        assert q["size"] == lim

    def test_limit_missing(self, search):
        params = {}

        search = query.Limiter()(search, params)
        q = search.to_dict()

        assert q["size"] == LIMIT_DEFAULT

    def test_sort_is_by_updated(self, search):
        """Sort defaults to "updated"."""
        params = {}

        search = query.Sorter()(search, params)
        q = search.to_dict()

        sort = q["sort"]
        assert len(sort) == 1
        assert list(sort[0].keys()) == ["updated"]

    def test_with_custom_sort(self, search):
        """Custom sorts are returned in the query dict."""
        params = {"sort": "title"}

        search = query.Sorter()(search, params)
        q = search.to_dict()

        assert q["sort"] == [{'title': {'unmapped_type': 'boolean', 'order': 'desc'}}]

    def test_order_defaults_to_desc(self, search):
        """'order': "desc" is returned in the q dict by default."""
        params = {}

        search = query.Sorter()(search, params)
        q = search.to_dict()

        sort = q["sort"]
        assert sort[0]["updated"]["order"] == "desc"

    def test_with_custom_order(self, search):
        """'order' params are returned in the query dict if given."""
        params = {"order": "asc"}

        search = query.Sorter()(search, params)
        q = search.to_dict()

        sort = q["sort"]
        assert sort[0]["updated"]["order"] == "asc"

    def test_defaults_to_match_all(self, search):
        """If no query params are given a "match_all": {} query is returned."""
        result = search.to_dict()

        assert result == {'query': {'match_all': {}}}

    def test_default_param_action(self, search):
        """Other params are added as "match" clauses."""
        params = {"foo": "bar"}

        search = query.KeyValueMatcher()(search, params)
        q = search.to_dict()

        assert q["query"] == {
            'bool': {'filter': [],
                     'must': [{'match': {'foo': 'bar'}}]},
        }

    def test_default_params_multidict(self, search):
        """Multiple params go into multiple "match" dicts."""
        params = multidict.MultiDict()
        params.add("user", "fred")
        params.add("user", "bob")

        search = query.KeyValueMatcher()(search, params)
        q = search.to_dict()

        assert q["query"] == {
            'bool': {'filter': [],
                     'must': [{'match': {'user': 'fred'}},
                              {'match': {'user': 'bob'}}]},
        }

    def test_with_evil_arguments(self, search):
        params = {
            "offset": "3foo",
            "limit": '\' drop table annotations'
        }

        search = query.Limiter()(search, params)
        q = search.to_dict()

        assert q["from"] == 0
        assert q["size"] == 20
        assert q["query"] == {'bool': {'filter': [], 'must': []}}


class TestSearch(object):
    def test_passes_params_to_matchers(self, search):
        testqualifier = mock.Mock()
        testqualifier.side_effect = lambda search, params: search
        search.append_qualifier(testqualifier)

        search.run({"foo": "bar"})

        testqualifier.assert_called_with(mock.ANY, {"foo": "bar"})

    def test_adds_qualifiers_to_query(self, search):
        testqualifier = mock.Mock()

        search.append_qualifier(testqualifier)

        assert testqualifier in search._qualifiers

    def test_passes_params_to_aggregations(self, search):
        testaggregation = mock.Mock()
        testaggregation.side_effect = lambda search, params: search
        search.append_aggregation(testaggregation)

        search.run({"foo": "bar"})

        testaggregation.assert_called_with(mock.ANY, {"foo": "bar"})

    def test_adds_aggregations_to_query(self, search):
        testaggregation = mock.Mock(key="foobar")

        search.append_aggregation(testaggregation)

        assert testaggregation in search._aggregations

    @pytest.fixture
    def search(self, pyramid_request):
        search = Search(pyramid_request)
        # Remove all default filters, aggregators, and matchers.
        search.clear()
        return search


def test_authority_filter_adds_authority_term(search):
    search = query.AuthorityFilter(authority='partner.org')(search, {})
    result = search.to_dict()["query"]["bool"]["filter"][0]
    assert result == {'term': {'authority': 'partner.org'}}


class TestAuthFilter(object):
    def test_unauthenticated(self, search):
        request = mock.Mock(authenticated_userid=None)
        search = query.AuthFilter(request)(search, {})
        result = search.to_dict()["query"]["bool"]["filter"][0]

        assert result == {'term': {'shared': True}}

    def test_authenticated(self, search):
        request = mock.Mock(authenticated_userid='acct:doe@example.org')
        search = query.AuthFilter(request)(search, {})
        result = search.to_dict()["query"]["bool"]["filter"][0]

        assert result == {
            'bool': {
                'should': [
                    {'term': {'shared': True}},
                    {'term': {'user_raw': 'acct:doe@example.org'}},
                ],
            }
        }


class TestGroupFilter(object):
    def test_term_filters_for_group(self, search):
        params = {"group": "wibble"}
        search = query.GroupFilter()(search, params)
        result = search.to_dict()["query"]["bool"]["filter"][0]

        assert result == {"term": {"group": "wibble"}}

    def test_strips_param(self, search):
        params = {"group": "wibble"}
        search = query.GroupFilter()(search, params)

        assert params == {}

    def test_returns_none_when_no_param(self, search):
        search = query.GroupFilter()(search, {})
        result = search.to_dict()

        assert result == {'query': {'match_all': {}}}


class TestGroupAuthFilter(object):
    def test_fetches_readable_groups(self, pyramid_request, group_service, search):
        pyramid_request.user = mock.sentinel.user

        search = query.GroupAuthFilter(pyramid_request)(search, {})

        group_service.groupids_readable_by.assert_called_once_with(mock.sentinel.user)

    def test_returns_terms_filter(self, pyramid_request, group_service, search):
        group_service.groupids_readable_by.return_value = ['group-a', 'group-b']

        search = query.GroupAuthFilter(pyramid_request)(search, {})
        result = search.to_dict()["query"]["bool"]["filter"][0]

        assert result == {'terms': {'group': ['group-a', 'group-b']}}


class TestUriFilter(object):
    @pytest.mark.usefixtures('uri')
    def test_inactive_when_no_uri_param(self, search):
        """
        When there's no `uri` parameter, return None.
        """
        request = mock.Mock()
        params = {}

        search = query.UriFilter(request)(search, params)
        result = search.to_dict()

        assert result == {'query': {'match_all': {}}}

    def test_expands_and_normalizes_into_terms_filter(self, storage, search):
        """
        Uses a `terms` filter against target.scope to filter for URI.

        UriFilter should use a `terms` filter against the normalized version of the
        target source field, which we store in `target.scope`.

        It should expand the input URI before searching, and normalize the results
        of the expansion.
        """
        request = mock.Mock()
        params = {"uri": "http://example.com/"}
        storage.expand_uri.side_effect = lambda _, x: [
            "http://giraffes.com/",
            "https://elephants.com/",
        ]

        search = query.UriFilter(request)(search, params)
        result = search.to_dict()["query"]["bool"]["filter"][0]
        query_uris = result["terms"]["target.scope"]

        storage.expand_uri.assert_called_with(request.db, "http://example.com/")
        assert sorted(query_uris) == sorted(["httpx://giraffes.com",
                                             "httpx://elephants.com"])

    def test_queries_multiple_uris(self, storage, search):
        """
        Uses a `terms` filter against target.scope to filter for URI.

        When multiple "uri" fields are supplied, the normalized URIs of all of
        them should be collected into a set and sent in the query.
        """
        request = mock.Mock()
        params = multidict.MultiDict()
        params.add("uri", "http://example.com")
        params.add("uri", "http://example.net")
        storage.expand_uri.side_effect = [
            ["http://giraffes.com/", "https://elephants.com/"],
            ["http://tigers.com/", "https://elephants.com/"],
        ]

        search = query.UriFilter(request)(search, params)
        result = search.to_dict()["query"]["bool"]["filter"][0]

        query_uris = result["terms"]["target.scope"]

        storage.expand_uri.assert_any_call(request.db, "http://example.com")
        storage.expand_uri.assert_any_call(request.db, "http://example.net")
        assert sorted(query_uris) == sorted(["httpx://giraffes.com",
                                             "httpx://elephants.com",
                                             "httpx://tigers.com"])

    def test_accepts_url_aliases(self, storage, search):
        request = mock.Mock()
        params = multidict.MultiDict()
        params.add("uri", "http://example.com")
        params.add("url", "http://example.net")
        storage.expand_uri.side_effect = [
            ["http://giraffes.com/", "https://elephants.com/"],
            ["http://tigers.com/", "https://elephants.com/"],
        ]

        search = query.UriFilter(request)(search, params)
        result = search.to_dict()["query"]["bool"]["filter"][0]

        query_uris = result["terms"]["target.scope"]

        storage.expand_uri.assert_any_call(request.db, "http://example.com")
        storage.expand_uri.assert_any_call(request.db, "http://example.net")
        assert sorted(query_uris) == sorted(["httpx://giraffes.com",
                                             "httpx://elephants.com",
                                             "httpx://tigers.com"])

    @pytest.fixture
    def storage(self, patch):
        storage = patch('h.search.query.storage')
        storage.expand_uri.side_effect = lambda x: [x]
        return storage

    @pytest.fixture
    def uri(self, patch):
        uri = patch('h.search.query.uri')
        uri.normalize.side_effect = lambda x: x
        return uri


class TestUserFilter(object):
    def test_term_filters_for_user(self, search):
        params = {"user": "luke"}
        search = query.UserFilter()(search, params)
        result = search.to_dict()["query"]["bool"]["filter"][0]

        assert result == {"terms": {"user": ["luke"]}}

    def test_supports_filtering_for_multiple_users(self, search):
        params = multidict.MultiDict()
        params.add("user", "alice")
        params.add("user", "luke")

        search = query.UserFilter()(search, params)
        result = search.to_dict()["query"]["bool"]["filter"][0]

        assert result == {
            "terms": {
                "user": ["alice", "luke"]
            }
        }

    def test_lowercases_value(self, search):
        params = {"user": "LUkE"}
        search = query.UserFilter()(search, params)
        result = search.to_dict()["query"]["bool"]["filter"][0]

        assert result == {"terms": {"user": ["luke"]}}

    def test_strips_param(self, search):
        params = {"user": "luke"}
        search = query.UserFilter()(search, params)

        assert params == {}

    def test_returns_none_when_no_param(self, search):
        params = {}
        search = query.UserFilter()(search, params)
        result = search.to_dict()

        assert result == {'query': {'match_all': {}}}


class TestDeletedFilter(object):
    def test_filter(self, search):
        search = query.DeletedFilter()(search, {})
        result = search.to_dict()["query"]["bool"]["filter"][0]

        assert result == {
            "bool": {"must_not": [{"exists": {"field": "deleted"}}]}
        }


class TestAnyMatcher():
    def test_any_query(self, search):
        params = {"any": "foo"}
        search = query.AnyMatcher()(search, params)
        result = search.to_dict()["query"]

        assert result == {
            "simple_query_string": {
                "fields": ["quote", "tags", "text", "uri.parts"],
                "query": "foo",
            }
        }

    def test_multiple_params(self, search):
        """Multiple keywords at once are handled correctly."""
        params = multidict.MultiDict()
        params.add("any", "howdy")
        params.add("any", "there")

        search = query.AnyMatcher()(search, params)
        result = search.to_dict()["query"]

        assert result == {
            "simple_query_string": {
                "fields": ["quote", "tags", "text", "uri.parts"],
                "query": "howdy there",
            }
        }


class TestTagsMatcher():
    def test_aliases_tag_to_tags(self, search):
        """'tag' params should be transformed into 'tags' queries.

        'tag' is aliased to 'tags' because users often type tag instead of tags.

        """
        params = multidict.MultiDict()
        params.add('tag', 'foo')
        params.add('tag', 'bar')

        search = query.TagsMatcher()(search, params)
        result = search.to_dict()["query"]

        assert list(result.keys()) == ['bool']
        assert list(result['bool'].keys()) == ['must']
        assert len(result['bool']['must']) == 2
        assert {'match': {'tags': {'query': 'foo', 'operator': 'and'}}} in result['bool']['must']
        assert {'match': {'tags': {'query': 'bar', 'operator': 'and'}}} in result['bool']['must']

    def test_with_both_tag_and_tags(self, search):
        """If both 'tag' and 'tags' params are used they should all become tags."""
        params = {'tag': 'foo', 'tags': 'bar'}

        search = query.TagsMatcher()(search, params)
        result = search.to_dict()["query"]

        assert list(result.keys()) == ['bool']
        assert list(result['bool'].keys()) == ['must']
        assert len(result['bool']['must']) == 2
        assert {'match': {'tags': {'query': 'foo', 'operator': 'and'}}} in result['bool']['must']
        assert {'match': {'tags': {'query': 'bar', 'operator': 'and'}}} in result['bool']['must']


class TestTagsAggregations(object):
    def test_key_is_tags(self):
        assert query.TagsAggregation().name == 'tags'

    def test_elasticsearch_aggregation(self, search):
        query.TagsAggregation()(search, {})
        agg = search.to_dict()["aggs"]["tags"]
        assert agg == {
            'terms': {'field': 'tags_raw', 'size': 10}
        }

    def test_it_allows_to_set_a_limit(self, search):
        query.TagsAggregation(limit=14)(search, {})
        agg = search.to_dict()["aggs"]["tags"]
        assert agg == {
            'terms': {'field': 'tags_raw', 'size': 14}
        }

    def test_parse_result(self):
        agg = query.TagsAggregation()
        elasticsearch_result = {
            "tags": {
                'buckets': [
                    {'key': 'tag-4', 'doc_count': 42},
                    {'key': 'tag-2', 'doc_count': 28},
                ]
            }
        }

        assert agg.parse_result(elasticsearch_result) == [
            {'tag': 'tag-4', 'count': 42},
            {'tag': 'tag-2', 'count': 28},
        ]


class TestUsersAggregation(object):
    def test_key_is_users(self):
        assert query.UsersAggregation().name == 'users'

    def test_elasticsearch_aggregation(self, search):
        query.UsersAggregation()(search, {})
        agg = search.to_dict()["aggs"]["users"]
        assert agg == {
            'terms': {'field': 'user_raw', 'size': 10}
        }

    def test_it_allows_to_set_a_limit(self, search):
        query.UsersAggregation(limit=14)(search, {})
        agg = search.to_dict()["aggs"]["users"]
        assert agg == {
            'terms': {'field': 'user_raw', 'size': 14}
        }

    def test_parse_result(self):
        agg = query.UsersAggregation()
        elasticsearch_result = {
            'users': {
                'buckets': [
                    {'key': 'alice', 'doc_count': 42},
                    {'key': 'luke', 'doc_count': 28},
                ]
            }
        }

        assert agg.parse_result(elasticsearch_result) == [
            {'user': 'alice', 'count': 42},
            {'user': 'luke', 'count': 28},
        ]


def test_nipsa_filter_filters_out_nipsad_annotations(group_service, pyramid_request, search):
    """nipsa_filter() filters out annotations with "nipsa": True."""
    pyramid_request.user = None
    search = query.NipsaFilter(pyramid_request)(search, {})
    filter_ = search.to_dict()["query"]["bool"]["filter"][0]
    assert filter_ == {
        "bool": {
            "should": [
                {'bool': {'must_not': [{'term': {'nipsa': True}}]}},
                {'exists': {'field': 'thread_ids'}},
            ]
        }
    }


def test_nipsa_filter_users_own_annotations_are_not_filtered(group_service, pyramid_request, search):
    search = query.NipsaFilter(pyramid_request)(search, {})
    filter_ = search.to_dict()["query"]["bool"]["filter"][0]

    assert {'term': {'user': 'fred'}} in (
        filter_["bool"]["should"])


def test_nipsa_filter_coerces_userid_to_lowercase(group_service, pyramid_request, user, search):
    user.userid = 'DonkeyNose'

    search = query.NipsaFilter(pyramid_request)(search, {})
    filter_ = search.to_dict()["query"]["bool"]["filter"][0]

    assert {'term': {'user': 'donkeynose'}} in (
        filter_["bool"]["should"])


def test_nipsa_filter_group_annotations_not_filtered_for_creator(group_service, pyramid_request, search):
    group_service.groupids_created_by.return_value = ['pubid-1', 'pubid-4', 'pubid-3']
    search = query.NipsaFilter(pyramid_request)(search, {})
    filter_ = search.to_dict()["query"]["bool"]["filter"][0]

    assert {'terms': {'group': ['pubid-1', 'pubid-4', 'pubid-3']}} in (
        filter_['bool']['should'])


@pytest.fixture
def search(pyramid_request, es_client):
    search = elasticsearch_dsl.Search(
            using=es_client.conn, index=pyramid_request.es.index
    )
    return search


@pytest.fixture
def pyramid_request(pyramid_request, user):
    pyramid_request.user = user
    return pyramid_request


@pytest.fixture
def user():
    return mock.Mock(userid='fred')
