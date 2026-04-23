from unittest.mock import MagicMock, patch

from chem2textqa.config.settings import Settings
from chem2textqa.scrapers.pubchem import PubChemScraper


def _make_settings() -> Settings:
    return Settings(ncbi_email="test@test.com")


def _mock_response(json_data, status_code=200):
    mock = MagicMock()
    mock.status_code = status_code
    mock.json.return_value = json_data
    return mock


@patch("chem2textqa.scrapers.pubchem.requests.Session")
def test_search_cids(mock_session_cls):
    session = MagicMock()
    mock_session_cls.return_value = session

    sdq_response = {
        "SDQOutputSet": [{
            "totalCount": 2,
            "rows": [{"cid": 2244}, {"cid": 5988}],
        }]
    }
    session.get.return_value = _mock_response(sdq_response)

    scraper = PubChemScraper(_make_settings())
    cids = scraper.search_cids("aspirin", max_results=10)

    assert cids == [2244, 5988]


@patch("chem2textqa.scrapers.pubchem.requests.Session")
def test_fetch_properties(mock_session_cls):
    session = MagicMock()
    mock_session_cls.return_value = session

    prop_response = {
        "PropertyTable": {
            "Properties": [
                {
                    "CID": 2244,
                    "Title": "Aspirin",
                    "MolecularFormula": "C9H8O4",
                    "MolecularWeight": 180.16,
                    "IUPACName": "2-acetyloxybenzoic acid",
                    "CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(O)=O",
                    "InChI": "InChI=1S/C9H8O4/...",
                    "InChIKey": "BSYNRYMUTXBXSQ-UHFFFAOYSA-N",
                },
            ]
        }
    }
    session.get.return_value = _mock_response(prop_response)

    scraper = PubChemScraper(_make_settings())
    props = scraper.fetch_properties([2244])

    assert 2244 in props
    assert props[2244]["Title"] == "Aspirin"
    assert props[2244]["MolecularFormula"] == "C9H8O4"


@patch("chem2textqa.scrapers.pubchem.requests.Session")
def test_fetch_linked_pmids_batch(mock_session_cls):
    session = MagicMock()
    mock_session_cls.return_value = session

    link_response = {
        "InformationList": {
            "Information": [
                {"CID": 2244, "PubMedID": [11111, 22222, 33333]},
                {"CID": 5988, "PubMedID": [44444]},
            ]
        }
    }
    session.get.return_value = _mock_response(link_response)

    scraper = PubChemScraper(_make_settings())
    result = scraper.fetch_linked_pmids_batch([2244, 5988])

    assert result[2244] == [11111, 22222, 33333]
    assert result[5988] == [44444]


@patch("chem2textqa.scrapers.pubchem.requests.Session")
def test_build_compound_record(mock_session_cls):
    mock_session_cls.return_value = MagicMock()

    scraper = PubChemScraper(_make_settings())
    record = scraper.build_compound_record(2244, {
        "Title": "Aspirin",
        "MolecularFormula": "C9H8O4",
        "MolecularWeight": 180.16,
        "CanonicalSMILES": "CC(=O)OC1=CC=CC=C1C(O)=O",
    })

    assert record.cid == 2244
    assert record.name == "Aspirin"
    assert record.molecular_formula == "C9H8O4"
    assert record.molecular_weight == 180.16


@patch("chem2textqa.scrapers.pubchem.requests.Session")
def test_collect_all_pmids(mock_session_cls):
    mock_session_cls.return_value = MagicMock()

    from chem2textqa.models.document import CompoundRecord

    scraper = PubChemScraper(_make_settings())
    records = [
        CompoundRecord(cid=1, name="A", linked_pmids=[100, 200, 300]),
        CompoundRecord(cid=2, name="B", linked_pmids=[200, 400]),
        CompoundRecord(cid=3, name="C", linked_pmids=[]),
    ]

    pmids = scraper.collect_all_pmids(records)
    assert pmids == [100, 200, 300, 400]  # sorted, deduplicated


@patch("chem2textqa.scrapers.pubchem.requests.Session")
def test_count_compounds(mock_session_cls):
    session = MagicMock()
    mock_session_cls.return_value = session

    session.get.return_value = _mock_response({
        "SDQOutputSet": [{"totalCount": 1785688}]
    })

    scraper = PubChemScraper(_make_settings())
    count = scraper.count_compounds("drug")
    assert count == 1785688
