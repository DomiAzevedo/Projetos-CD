import docker
from vespa.io import VespaResponse, VespaQueryResponse
from vespa.package import (
    ApplicationPackage,
    Field,
    Schema,
    Document,
    HNSW,
    RankProfile,
    Component,
    Parameter,
    FieldSet,
    GlobalPhaseRanking,
    Function,
)
from vespa.deployment import VespaDocker
import pandas as pd
import numpy as np

class VespaApp:
    def __init__(self):
        self.app = self.vespa_docker_deploy()
        self.app = self.start_vespa(self.app)

    def create_package(self):
        package = ApplicationPackage(
            name="hybridsearchbv",
            schema=[
                Schema(
                    name="doc",
                    document=Document(
                        fields=[
                            Field(name="id", type="string", indexing=["summary"]),
                            Field(
                                name="title",
                                type="string",
                                indexing=["index", "summary"],
                                index="enable-bm25",
                            ),
                            Field(
                                name="authors",
                                type="string",
                                indexing=["index", "summary"],
                                index="enable-bm25",
                                bolding=True,
                            ),
                            Field(
                                name="description",
                                type="string",
                                indexing=["index", "summary"],
                                index="enable-bm25",
                                bolding=True,
                            ),
                            Field(
                                name="categories",
                                type="string",
                                indexing=["index", "summary"],
                                index="enable-bm25",
                                bolding=True,
                            ),
                            Field(
                                name="embedding",
                                type="tensor<float>(x[384])",
                                indexing=[
                                    'input title . " " . input authors . " " . input description . " " . input categories',
                                    "embed",
                                    "index",
                                    "attribute",
                                ],
                                ann=HNSW(distance_metric="angular"),
                                is_document_field=False,
                            ),
                        ]
                    ),
                    fieldsets=[FieldSet(name="default", fields=["title", "authors", "description", "categories"])],
                    rank_profiles=[
                        RankProfile(
                            name="bm25",
                            inputs=[("query(q)", "tensor<float>(x[384])")],
                            functions=[
                                Function(name="bm25sum", expression="bm25(title) + bm25(authors) + bm25(description) + bm25(categories)")
                            ],
                            first_phase="bm25sum",
                        ),
                        RankProfile(
                            name="semantic",
                            inputs=[("query(q)", "tensor<float>(x[384])")],
                            first_phase="closeness(field, embedding)",
                        ),
                        RankProfile(
                            name="fusion",
                            inherits="bm25",
                            inputs=[("query(q)", "tensor<float>(x[384])")],
                            first_phase="closeness(field, embedding)",
                            global_phase=GlobalPhaseRanking(
                                expression="reciprocal_rank_fusion(bm25sum, closeness(field, embedding))",
                                rerank_count=1000,
                            ),
                        ),
                    ],
                )
            ],
            components=[
                Component(
                    id="e5",
                    type="hugging-face-embedder",
                    parameters=[
                        Parameter(
                            "transformer-model",
                            {
                                "url": "https://github.com/vespa-engine/sample-apps/raw/master/simple-semantic-search/model/e5-small-v2-int8.onnx"
                            },
                        ),
                        Parameter(
                            "tokenizer-model",
                            {
                                "url": "https://raw.githubusercontent.com/vespa-engine/sample-apps/master/simple-semantic-search/model/tokenizer.json"
                            },
                        ),
                    ],
                )
            ],
        )

        return package


    def vespa_docker_deploy(self):
        package = self.create_package()

        vespa_docker = VespaDocker()
        app = vespa_docker.deploy(application_package=package)

        return app


    def transform_row(self, row):
        return {
            "id": row["id"],
            "fields": {"title": row["title"], "authors": row["authors"], "description": row["description"], "categories": row["categories"], "id": row["id"]},
        }


    def callback(self, response:VespaResponse, id:str):
        if not response.is_successful():
            print(f"Error when feeding document {id}: {response.get_json()}")


    def hits_as_df(self, response:VespaQueryResponse, fields) -> pd.DataFrame:
        records = []
        for hit in response.hits:
            record = {}
            for field in fields:
                record[field] = hit['fields'][field]
            records.append(record)
        return pd.DataFrame(records)


    def start_vespa(self, app):
        df = pd.read_csv("https://raw.githubusercontent.com/bernardovma/dados_livros/main/data.csv")
        df['id'] = range(1, len(df) + 1)
        df = df.fillna("")
        vespa_feed = df.apply(self.transform_row, axis=1).tolist()

        app.feed_iterable(vespa_feed, schema="doc", namespace="bookrec", callback=self.callback)

        return app


    def query_bm25(self, query, limit = 10):
        with self.app.syncio(connections=12) as session:
                response:VespaQueryResponse = session.query(
                    yql=f"select * from sources * where userQuery() limit {limit}",
                    query=query,
                    ranking="bm25"
                )
                assert(response.is_successful())

        return self.hits_as_df(response, ['id', 'title', 'authors', 'description', 'categories'])


    def query_semantic(self, input_query):
        with self.app.syncio(connections=12) as session:
            query = input_query
            response: VespaQueryResponse = session.query(
                yql="select * from sources * where ({targetHits:1000}nearestNeighbor(embedding,q)) limit 10",
                query=query,
                ranking="semantic",
                body={"input.query(q)": f"embed({query})"},
            )
            assert response.is_successful()
        return self.hits_as_df(response, ['id', 'title'])
    

    def query_hybrid(self, input_query):
        with self.app.syncio(connections=12) as session:
            query = input_query
            response: VespaQueryResponse = session.query(
                yql="select * from sources * where userQuery() or ({targetHits:1000}nearestNeighbor(embedding,q)) limit 10",
                query=query,
                ranking="fusion",
                body={"input.query(q)": f"embed({query})"},
            )
            assert response.is_successful()

        return self.hits_as_df(response, ["id", "title"]) 