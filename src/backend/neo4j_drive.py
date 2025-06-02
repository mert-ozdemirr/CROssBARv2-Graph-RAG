from neo4j import GraphDatabase

def neo4j_driver_set():
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "crossbar"
    driver_n4j = GraphDatabase.driver(uri, auth=(username, password))
    return driver_n4j

