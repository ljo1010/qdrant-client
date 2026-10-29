import numpy as np
import pytest

from qdrant_client.client_base import QdrantBase
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import models
from tests.congruence_tests.test_common import (
    COLLECTION_NAME,
    compare_client_results,
    generate_sparse_fixtures,
    init_client,
    init_local,
    init_remote,
    initialize_fixture_collection,
    sparse_image_vector_size,
    sparse_text_vector_size,
    sparse_vectors_config,
)
from tests.fixtures.filters import one_random_filter_please
from tests.fixtures.points import generate_random_sparse_vector, random_sparse_vectors


class TestSimpleSparseSearcher:
    __test__ = False

    def __init__(self):
        self.query_text = generate_random_sparse_vector(sparse_text_vector_size, density=0.3)
        self.query_image = generate_random_sparse_vector(sparse_image_vector_size, density=0.2)

    def simple_search_text(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            using="sparse-text",
            query=self.query_text,
            with_payload=True,
            with_vectors=["sparse-text"],
            limit=10,
        ).points

    def simple_search_image(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_image,
            using="sparse-image",
            with_payload=True,
            with_vectors=["sparse-image"],
            limit=10,
        ).points

    def simple_search_text_offset(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=self.query_text,
            using="sparse-text",
            with_payload=True,
            limit=10,
            offset=10,
        ).points

    def search_score_threshold(self, client: QdrantBase) -> list[models.ScoredPoint]:
        res1 = client.query_points(
            collection_name=COLLECTION_NAME,
            using="sparse-text",
            query=self.query_text,
            with_payload=True,
            limit=10,
            score_threshold=0.9,
        ).points

        res2 = client.query_points(
            collection_name=COLLECTION_NAME,
            using="sparse-text",
            query=self.query_text,
            with_payload=True,
            limit=10,
            score_threshold=0.95,
        ).points

        res3 = client.query_points(
            collection_name=COLLECTION_NAME,
            using="sparse-text",
            query=self.query_text,
            with_payload=True,
            limit=10,
            score_threshold=0.1,
        ).points

        return res1 + res2 + res3

    def simple_search_text_select_payload(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            using="sparse-text",
            query=self.query_text,
            with_payload=["text_array", "nested.id"],
            limit=10,
        ).points

    def search_payload_exclude(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            using="sparse-text",
            query=self.query_text,
            with_payload=models.PayloadSelectorExclude(exclude=["text_array", "nested.id"]),
            limit=10,
        ).points

    def simple_search_image_select_vector(self, client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            using="sparse-image",
            query=self.query_image,
            with_payload=False,
            with_vectors=["sparse-image", "sparse-text"],
            limit=10,
        ).points

    def filter_search_text(
        self, client: QdrantBase, query_filter: models.Filter
    ) -> list[models.ScoredPoint]:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            using="sparse-text",
            query=self.query_text,
            query_filter=query_filter,
            with_payload=True,
            limit=10,
        ).points

    def default_mmr_query(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.query_text,
                mmr=models.Mmr(),
            ),
            using="sparse-text",
            limit=10,
        )

    def mmr_query_parametrized(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.query_text,
                mmr=models.Mmr(diversity=0.3, candidates_limit=30),
            ),
            using="sparse-text",
            limit=10,
        )

    def mmr_query_parametrized_score_threshold(self, client: QdrantBase) -> models.QueryResponse:
        return client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.NearestQuery(
                nearest=self.query_text,
                mmr=models.Mmr(diversity=0.3, candidates_limit=30),
            ),
            score_threshold=3.3,
            using="sparse-text",
            limit=10,
        )


def test_simple_search():
    fixture_points = generate_sparse_fixtures()

    searcher = TestSimpleSparseSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    compare_client_results(local_client, remote_client, searcher.simple_search_text)
    compare_client_results(local_client, remote_client, searcher.simple_search_image)
    compare_client_results(local_client, remote_client, searcher.simple_search_text_offset)
    compare_client_results(local_client, remote_client, searcher.search_score_threshold)
    compare_client_results(local_client, remote_client, searcher.simple_search_text_select_payload)
    compare_client_results(local_client, remote_client, searcher.simple_search_image_select_vector)
    compare_client_results(local_client, remote_client, searcher.search_payload_exclude)

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client, remote_client, searcher.filter_search_text, query_filter=query_filter
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_mmr():
    fixture_points = generate_sparse_fixtures(num=100)

    searcher = TestSimpleSparseSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    compare_client_results(local_client, remote_client, searcher.default_mmr_query)
    compare_client_results(local_client, remote_client, searcher.mmr_query_parametrized)
    compare_client_results(
        local_client, remote_client, searcher.mmr_query_parametrized_score_threshold
    )


def test_simple_opt_vectors_search():
    fixture_points = generate_sparse_fixtures(skip_vectors=True)

    searcher = TestSimpleSparseSearcher()

    local_client = init_local()
    init_client(local_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    remote_client = init_remote()
    init_client(remote_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

    compare_client_results(local_client, remote_client, searcher.simple_search_text)
    compare_client_results(local_client, remote_client, searcher.simple_search_image)
    compare_client_results(local_client, remote_client, searcher.simple_search_text_offset)
    compare_client_results(local_client, remote_client, searcher.search_score_threshold)
    compare_client_results(local_client, remote_client, searcher.simple_search_text_select_payload)
    compare_client_results(local_client, remote_client, searcher.simple_search_image_select_vector)
    compare_client_results(local_client, remote_client, searcher.search_payload_exclude)

    for i in range(100):
        query_filter = one_random_filter_please()
        try:
            compare_client_results(
                local_client, remote_client, searcher.filter_search_text, query_filter=query_filter
            )
        except AssertionError as e:
            print(f"\nFailed with filter {query_filter}")
            raise e


def test_search_with_persistence():
    import tempfile

    fixture_points = generate_sparse_fixtures()
    searcher = TestSimpleSparseSearcher()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_client = init_local(tmpdir)
        init_client(local_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

        payload_update_filter = one_random_filter_please()
        local_client.set_payload(COLLECTION_NAME, {"test": f"test"}, payload_update_filter)

        local_client.close()
        local_client_2 = init_local(tmpdir)

        remote_client = init_remote()
        init_client(remote_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

        remote_client.set_payload(COLLECTION_NAME, {"test": f"test"}, payload_update_filter)

        payload_update_filter = one_random_filter_please()
        local_client_2.set_payload(COLLECTION_NAME, {"test": "test2"}, payload_update_filter)
        remote_client.set_payload(COLLECTION_NAME, {"test": "test2"}, payload_update_filter)

        for i in range(10):
            query_filter = one_random_filter_please()
            try:
                compare_client_results(
                    local_client_2,
                    remote_client,
                    searcher.filter_search_text,
                    query_filter=query_filter,
                )
            except AssertionError as e:
                print(f"\nFailed with filter {query_filter}")
                raise e


def test_search_with_persistence_and_skipped_vectors():
    import tempfile

    fixture_points = generate_sparse_fixtures(skip_vectors=True)
    searcher = TestSimpleSparseSearcher()
    with tempfile.TemporaryDirectory() as tmpdir:
        local_client = init_local(tmpdir)
        init_client(local_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

        payload_update_filter = one_random_filter_please()
        local_client.set_payload(COLLECTION_NAME, {"test": f"test"}, payload_update_filter)

        count_before_load = local_client.count(COLLECTION_NAME)
        local_client.close()
        local_client_2 = init_local(tmpdir)

        count_after_load = local_client_2.count(COLLECTION_NAME)

        assert count_after_load == count_before_load

        remote_client = init_remote()
        init_client(remote_client, fixture_points, sparse_vectors_config=sparse_vectors_config)

        remote_client.set_payload(COLLECTION_NAME, {"test": f"test"}, payload_update_filter)

        payload_update_filter = one_random_filter_please()
        local_client_2.set_payload(COLLECTION_NAME, {"test": "test2"}, payload_update_filter)
        remote_client.set_payload(COLLECTION_NAME, {"test": "test2"}, payload_update_filter)

        for i in range(10):
            query_filter = one_random_filter_please()
            try:
                compare_client_results(
                    local_client_2,
                    remote_client,
                    searcher.filter_search_text,
                    query_filter=query_filter,
                )
            except AssertionError as e:
                print(f"\nFailed with filter {query_filter}")
                raise e


def test_query_with_nan():
    local_client = init_local()
    remote_client = init_remote()

    fixture_points = generate_sparse_fixtures()
    sparse_vector = random_sparse_vectors({"sparse-text": sparse_text_vector_size})

    sparse_vector["sparse-text"].values[0] = np.nan

    local_client.create_collection(
        COLLECTION_NAME, vectors_config={}, sparse_vectors_config=sparse_vectors_config
    )
    if remote_client.collection_exists(COLLECTION_NAME):
        remote_client.delete_collection(COLLECTION_NAME)
    remote_client.create_collection(
        COLLECTION_NAME, vectors_config={}, sparse_vectors_config=sparse_vectors_config
    )
    init_client(
        local_client,
        fixture_points,
        vectors_config={},
        sparse_vectors_config=sparse_vectors_config,
    )
    init_client(
        remote_client,
        fixture_points,
        vectors_config={},
        sparse_vectors_config=sparse_vectors_config,
    )

    with pytest.raises(ValueError):
        local_client.query_points(
            COLLECTION_NAME, sparse_vector["sparse-text"], using="sparse-text"
        )
    with pytest.raises(UnexpectedResponse):
        remote_client.query_points(
            COLLECTION_NAME, sparse_vector["sparse-text"], using="sparse-text"
        )


ZERO_SCORE_QUERY = models.SparseVector(indices=[1, 2], values=[1.0, 1.0])
ZERO_SCORE_NUM_POINTS = 60


def init_zero_score_collection(
    client: QdrantBase, num_in: int, sparse_params: models.SparseVectorParams
) -> None:
    """Point 1 scores exactly 0.0 against `ZERO_SCORE_QUERY`, the others 2.0.

    Points 1..num_in have tag="in", the rest tag="out".
    """
    initialize_fixture_collection(
        client, vectors_config={}, sparse_vectors_config={"sparse-text": sparse_params}
    )
    client.create_payload_index(COLLECTION_NAME, "tag", models.PayloadSchemaType.KEYWORD)
    client.upsert(
        COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=i,
                vector={
                    "sparse-text": models.SparseVector(
                        indices=[1, 2], values=[1.0, -1.0] if i == 1 else [1.0, 1.0]
                    )
                },
                payload={"tag": "in" if i <= num_in else "out"},
            )
            for i in range(1, ZERO_SCORE_NUM_POINTS + 1)
        ],
        wait=True,
    )


@pytest.mark.parametrize("modifier", [None, models.Modifier.IDF])
def test_zero_score_nearest_search(modifier: models.Modifier | None) -> None:
    # The server drops exact 0.0 scores on its inverted index path: no filter, or a filter
    # matching at least `full_scan_threshold` points. Smaller filters take the plain path,
    # which keeps them.
    in_filter = models.Filter(
        must=[models.FieldCondition(key="tag", match=models.MatchValue(value="in"))]
    )

    def search(
        client: QdrantBase, query_filter: models.Filter | None = None
    ) -> list[models.ScoredPoint]:
        return client.query_points(
            COLLECTION_NAME,
            query=ZERO_SCORE_QUERY,
            using="sparse-text",
            query_filter=query_filter,
            limit=ZERO_SCORE_NUM_POINTS,
        ).points

    local_client = init_local()
    remote_client = init_remote()
    sparse_params = models.SparseVectorParams(
        index=models.SparseIndexParams(full_scan_threshold=10), modifier=modifier
    )

    for num_in in (5, 60):
        for client in (local_client, remote_client):
            init_zero_score_collection(client, num_in, sparse_params)

        compare_client_results(local_client, remote_client, search)
        compare_client_results(local_client, remote_client, search, query_filter=in_filter)

        assert 1 not in [point.id for point in search(local_client)]
        filtered_ids = [point.id for point in search(local_client, in_filter)]
        assert (1 in filtered_ids) == (num_in < 10)


def test_zero_score_nearest_groups() -> None:
    # The server only takes points with the `group_by` key as group candidates. That filter
    # matches fewer points than the default `full_scan_threshold`, so 0.0 scores are kept.
    def query_groups(client: QdrantBase) -> models.GroupsResult:
        return client.query_points_groups(
            COLLECTION_NAME,
            query=ZERO_SCORE_QUERY,
            using="sparse-text",
            group_by="tag",
            limit=2,
            group_size=ZERO_SCORE_NUM_POINTS,
        )

    local_client = init_local()
    remote_client = init_remote()
    for client in (local_client, remote_client):
        init_zero_score_collection(client, 5, models.SparseVectorParams())

    compare_client_results(local_client, remote_client, query_groups)

    hit_ids = [hit.id for group in query_groups(local_client).groups for hit in group.hits]
    assert 1 in hit_ids


def test_zero_score_recommend_and_context() -> None:
    # Unlike nearest search, the server scores recommend and context queries against every
    # point and keeps 0.0 scores.
    def recommend_sum_scores(client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            COLLECTION_NAME,
            query=models.RecommendQuery(
                recommend=models.RecommendInput(
                    positive=[1], strategy=models.RecommendStrategy.SUM_SCORES
                )
            ),
            using="sparse-text",
            limit=10,
        ).points

    def context(client: QdrantBase) -> list[models.ScoredPoint]:
        return client.query_points(
            COLLECTION_NAME,
            query=models.ContextQuery(context=[models.ContextPair(positive=2, negative=3)]),
            using="sparse-text",
            limit=10,
        ).points

    vectors = {
        1: models.SparseVector(indices=[1], values=[1.0]),
        2: models.SparseVector(indices=[1], values=[1.0]),
        3: models.SparseVector(indices=[9], values=[1.0]),
        4: models.SparseVector(indices=[1, 2], values=[1.0, -1.0]),
    }
    local_client = init_local()
    remote_client = init_remote()
    for client in (local_client, remote_client):
        initialize_fixture_collection(
            client, vectors_config={}, sparse_vectors_config=sparse_vectors_config
        )
        client.upsert(
            COLLECTION_NAME,
            points=[
                models.PointStruct(id=id_, vector={"sparse-text": vector})
                for id_, vector in vectors.items()
            ],
            wait=True,
        )

    compare_client_results(local_client, remote_client, recommend_sum_scores)
    compare_client_results(local_client, remote_client, context, is_context_search=True)

    assert (3, 0.0) in [(point.id, point.score) for point in recommend_sum_scores(local_client)]
    assert sorted((point.id, point.score) for point in context(local_client)) == [
        (1, 0.0),
        (4, 0.0),
    ]
