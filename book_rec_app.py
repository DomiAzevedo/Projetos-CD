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
    FirstPhaseRanking,
    SecondPhaseRanking
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
            name="bookrecapp",
            schema=[
                Schema(
                    name="doc",
                    document=Document(
                        fields=[
                            Field(name="id", type="string", indexing=["summary"]),
                            Field(name="title", type="string", indexing=["index", "summary"]),
                            Field(
                                name="authors",
                                type="string",
                                indexing=["index", "summary"],
                                bolding=False,
                            ),
                            Field(
                                name="categories",
                                type="string",
                                indexing=["index", "summary"],
                                bolding=False,
                            ),
                            Field(name="description", type="array<string>", indexing=["summary", "index"]),
                            Field(
                                name="embedding",
                                type="tensor<float>(x[384])",
                                indexing=[
                                    'input description . " " . input categories',
                                    "embed e5",
                                    "index",
                                    "attribute",
                                ],
                                ann=HNSW(distance_metric="angular"),
                                is_document_field=False,
                            ),
                            Field(
                                name="colbert",
                                type="tensor<int8>(description{}, token{}, v[16])",
                                indexing=["input description", "embed colbert description", "attribute"],
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
                                Function(name="bm25sum", expression="bm25(description) + bm25(categories)")
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
                        RankProfile(
                            name="bm25_semantic",
                            inputs=[("query(q)", "tensor<float>(x[384])")],
                            functions=[
                                Function(name="bm25sum", expression="bm25(description) + bm25(categories)"),
                                Function(name="closeness", expression="closeness(field, embedding)"),
                            ],
                            first_phase=FirstPhaseRanking(expression = "bm25sum"),
                            second_phase=SecondPhaseRanking(expression = "closeness", rerank_count=1000),
                            match_features=["bm25sum", "closeness"],
                        ),
                        RankProfile(
                            name="colbert_local",
                            inputs=[
                                ("query(q)", "tensor<float>(x[384])"),
                                ("query(qt)", "tensor<float>(querytoken{}, v[128])"),
                            ],
                            functions=[
                                Function(name="cos_sim", expression="closeness(field, embedding)"),
                                Function(
                                    name="max_sim_per",
                                    expression="""
                                        sum(
                                            reduce(
                                                sum(
                                                    query(qt) * unpack_bits(attribute(colbert)) , v
                                                ),
                                                max, token
                                            ),
                                            querytoken
                                        )
                                    """,
                                ),
                                Function(
                                    name="max_sim_local", expression="reduce(max_sim_per, max, description)"
                                ),
                            ],
                            first_phase=FirstPhaseRanking(expression="cos_sim"),
                            second_phase=SecondPhaseRanking(expression="max_sim_local"),
                            match_features=["cos_sim", "max_sim_local", "max_sim_per"],
                        ),
                        RankProfile(
                            name="colbert_global",
                            inputs=[
                                ("query(q)", "tensor<float>(x[384])"),
                                ("query(qt)", "tensor<float>(querytoken{}, v[128])"),
                            ],
                            functions=[
                                Function(name="cos_sim", expression="closeness(field, embedding)"),
                                Function(
                                    name="max_sim_cross",
                                    expression="""
                                        sum(
                                            reduce(
                                                sum(
                                                    query(qt) *  unpack_bits(attribute(colbert)) , v
                                                ),
                                                max, token, description
                                            ),
                                            querytoken
                                        )
                                        """
                                ),
                                Function(
                                    name="max_sim_global", expression="reduce(max_sim_cross, max)"
                                ),
                            ],
                            first_phase=FirstPhaseRanking(expression="cos_sim"),
                            second_phase=SecondPhaseRanking(expression="max_sim_global", rerank_count=1000),
                            match_features=[
                            "cos_sim",
                            "max_sim_global",
                            "max_sim_cross",
                            ],
                        ),
                        RankProfile(
                            name="bm25_colbert",
                            inputs=[
                                ("query(q)", "tensor<float>(x[384])"),
                                ("query(qt)", "tensor<float>(querytoken{}, v[128])"),
                            ],
                            functions=[
                                Function(name="bm25sum", expression="bm25(description) + bm25(categories)"),
                                Function(name = "closeness", expression = "closeness(field, embedding)"),
                                Function(
                                    name="max_sim_per",
                                    expression="""
                                        sum(
                                            reduce(
                                                sum(
                                                    query(qt) * unpack_bits(attribute(colbert)) , v
                                                ),
                                                max, token
                                            ),
                                            querytoken
                                        )
                                    """,
                                ),
                                Function(
                                    name="max_sim_local", expression="reduce(max_sim_per, max, description)"
                                ),
                            ],
                            first_phase=FirstPhaseRanking(expression = "bm25sum + closeness(field, embedding)"),
                            second_phase=SecondPhaseRanking(expression = "max_sim_local", rerank_count=500),
                            match_features=["bm25sum", "max_sim_per", "max_sim_local"],
                        ),
                        RankProfile(
                            name="bm25_colbert_global",
                            inputs=[
                                ("query(q)", "tensor<float>(x[384])"),
                                ("query(qt)", "tensor<float>(querytoken{}, v[128])"),
                            ],
                            functions=[
                                Function(name="bm25sum", expression="bm25(description) + bm25(categories)"),
                                Function(name = "closeness", expression = "closeness(field, embedding)"),
                                Function(
                                    name="max_sim_cross",
                                    expression="""
                                        sum(
                                            reduce(
                                                sum(
                                                    query(qt) *  unpack_bits(attribute(colbert)) , v
                                                ),
                                                max, token, description
                                            ),
                                            querytoken
                                        )
                                        """
                                ),
                                Function(
                                    name="max_sim_global", expression="reduce(max_sim_cross, max)"
                                ),
                            ],
                            first_phase=FirstPhaseRanking(expression = "bm25sum + closeness(field, embedding)"),
                            second_phase=SecondPhaseRanking(expression = "max_sim_global", rerank_count=500),
                            match_features=["bm25sum", "max_sim_cross", "max_sim_global"],
                        ),
                    ],
                ),
            ],
            components=[
                Component(
                    id="e5",
                    type="hugging-face-embedder",
                    parameters=[
                        Parameter(
                            name="transformer-model",
                            args={
                                "url": "https://huggingface.co/intfloat/e5-small-v2/resolve/main/model.onnx"
                            },
                        ),
                        Parameter(
                            name="tokenizer-model",
                            args={
                                "url": "https://huggingface.co/intfloat/e5-small-v2/raw/main/tokenizer.json"
                            },
                        ),
                    ],
                ),
                Component(
                    id="colbert",
                    type="colbert-embedder",
                    parameters=[
                        Parameter(
                            name="transformer-model",
                            args={
                                "url": "https://huggingface.co/colbert-ir/colbertv2.0/resolve/main/model.onnx"
                            },
                        ),
                        Parameter(
                            name="tokenizer-model",
                            args={
                                "url": "https://huggingface.co/colbert-ir/colbertv2.0/raw/main/tokenizer.json"
                            },
                        ),
                    ],
                ),
            ]
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


    def hits_as_df(self, response, fields):
        records = []
        for hit in response.hits:
            record = {}
            for field in fields:
                record[field] = hit['fields'].get(field, None)  
            records.append(record)
        return pd.DataFrame(records)


    def start_vespa(self, app):
        df = pd.read_csv("https://raw.githubusercontent.com/bernardovma/dados_livros/main/data.csv")
        df['id'] = range(1, len(df) + 1)
        df = df.fillna("")
        df['description'] = df['description'].apply(lambda x: [x])
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
        return self.hits_as_df(response, ['id', 'title', 'authors', 'description', 'categories'])
    

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

        return self.hits_as_df(response, ['id', 'title', 'authors', 'description', 'categories']) 
    
    def query_second_phase(self, input_query):
        with self.app.syncio(connections=25) as session:
            query = input_query
            response: VespaQueryResponse = session.query(
                yql="select * from sources * where userQuery() or ({targetHits:1000}nearestNeighbor(embedding,q)) limit 10",
                query=query,
                ranking="bm25_semantic",
                body={"input.query(q)": f"embed(e5, '{query}')"},
            )
            assert response.is_successful()

        return self.hits_as_df(response, ['id', 'title', 'authors', 'description', 'categories']) 

    # def query_second_phase_colbert(self, input_query):
    #     with self.app.syncio(connections=25) as session:
    #         query = input_query
    #         response: VespaQueryResponse = session.query(
    #             yql="select * from sources * where userQuery() or ({targetHits:1000}nearestNeighbor(embedding,q)) limit 10",
    #             query=query,
    #             ranking="bm25_semantic",
    #             body={
    #                 "input.query(q)": f'embed(e5, "{query}")',
    #                 "input.query(qt)": f'embed(colbert, "{query}")'
    #             },
    #         )
    #         assert response.is_successful()

    #     return self.hits_as_df(response, ['id', 'title', 'authors', 'description', 'categories']) 
    
    def query_colbert(self, input_query):
        with self.app.syncio(connections=25) as session:
            query = input_query
            response: VespaQueryResponse = session.query(
                yql="select * from sources * where userQuery() or ({targetHits:1000}nearestNeighbor(embedding,q)) limit 10",
                query=query,
                ranking="colbert_local",
                body={
                    "input.query(q)": f'embed(e5, "{query}")',
                    "input.query(qt)": f'embed(colbert, "{query}")'
                },
            )
            assert response.is_successful()

        return self.hits_as_df(response, ['id', 'title', 'authors', 'description', 'categories'])


    def query_colbert_global(self, input_query):
        with self.app.syncio(connections=25) as session:
            query = input_query
            response: VespaQueryResponse = session.query(
                yql="select * from sources * where userQuery() or ({targetHits:1000}nearestNeighbor(embedding,q)) limit 10",
                query=query,
                ranking="colbert_global",
                body={
                    "input.query(q)": f'embed(e5, "{query}")',
                    "input.query(qt)": f'embed(colbert, "{query}")'
                },
            )
            assert response.is_successful()

        return self.hits_as_df(response, ['id', 'title', 'authors', 'description', 'categories'])
    
    def query_colbert_2phase(self, input_query):
        with self.app.syncio(connections=25) as session:
            query = input_query
            response: VespaQueryResponse = session.query(
                yql="select * from sources * where userQuery() or ({targetHits:1000}nearestNeighbor(embedding,q)) limit 10",
                query=query,
                ranking="bm25_colbert",
                body={
                    "input.query(q)": f'embed(e5, "{query}")',
                    "input.query(qt)": f'embed(colbert, "{query}")'
                },
            )
            assert response.is_successful()

        return self.hits_as_df(response, ['id', 'title', 'authors', 'description', 'categories'])

    def query_colbert_2phase_global(self, input_query):
        with self.app.syncio(connections=25) as session:
            query = input_query
            response: VespaQueryResponse = session.query(
                yql="select * from sources * where userQuery() or ({targetHits:1000}nearestNeighbor(embedding,q)) limit 10",
                query=query,
                ranking="bm25_colbert_global",
                body={
                    "input.query(q)": f'embed(e5, "{query}")',
                    "input.query(qt)": f'embed(colbert, "{query}")'
                },
            )
            assert response.is_successful()

        return self.hits_as_df(response, ['id', 'title', 'authors', 'description', 'categories'])