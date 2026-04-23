import gzip
import json
from pathlib import Path

from chem2textqa.processing.compounds import (
    build_pmid_to_cids,
    load_compound_index,
)
from chem2textqa.processing.mesh_local import (
    LOCAL_QA_CATEGORIES,
    categorize_mesh_headings,
)
from chem2textqa.processing.pubmed_xml import iter_pubmed_articles


# ------------------------------------------------------------------
# MeSH local categorizer
# ------------------------------------------------------------------


def test_categorize_pharmacology():
    headings = ["Aspirin/*pharmacology", "Humans"]
    result = categorize_mesh_headings(headings)
    assert "mechanism_of_action" in result


def test_categorize_therapeutic_use():
    headings = ["Metformin/therapeutic use", "Diabetes Mellitus, Type 2"]
    result = categorize_mesh_headings(headings)
    assert "therapeutic_use" in result


def test_categorize_multiple():
    headings = [
        "Aspirin/*pharmacology",
        "Aspirin/therapeutic use",
        "Aspirin/adverse effects",
        "Humans",
    ]
    result = categorize_mesh_headings(headings)
    assert "mechanism_of_action" in result
    assert "therapeutic_use" in result
    assert "toxicity" in result


def test_categorize_drug_interactions():
    headings = ["Drug Interactions", "Warfarin/pharmacology"]
    result = categorize_mesh_headings(headings)
    assert "drug_interactions" in result
    assert "mechanism_of_action" in result


def test_categorize_metabolism():
    headings = ["Cytochrome P-450 CYP3A/metabolism"]
    result = categorize_mesh_headings(headings)
    assert "metabolism" in result


def test_categorize_no_match():
    headings = ["Humans", "Male", "Adult"]
    result = categorize_mesh_headings(headings)
    assert result == []


def test_categorize_empty():
    assert categorize_mesh_headings([]) == []


def test_qa_categories_present():
    names = {c.name for c in LOCAL_QA_CATEGORIES}
    assert "mechanism_of_action" in names
    assert "therapeutic_use" in names
    assert "toxicity" in names
    assert "metabolism" in names
    assert "drug_interactions" in names
    assert "chemistry" in names


# ------------------------------------------------------------------
# Compound index loading
# ------------------------------------------------------------------


def _make_bulk_fixture(tmp_path: Path) -> Path:
    """Create a tiny fake bulk directory with CID-* files."""
    bulk = tmp_path / "bulk"
    bulk.mkdir()

    # CID-MeSH (defines target set)
    (bulk / "CID-MeSH").write_text(
        "2244\tAspirin\tAcetylsalicylic Acid\n"
        "4091\tMetformin\n"
        "5743\tMorphine\n"
    )

    def _gz(name: str, content: str) -> None:
        with gzip.open(bulk / name, "wt") as f:
            f.write(content)

    _gz("CID-Title.gz",
        "1\tMethane\n"
        "2244\tAspirin\n"
        "4091\tMetformin\n"
        "5743\tMorphine\n"
        "9999\tIrrelevant Compound\n"
    )
    _gz("CID-SMILES.gz",
        "2244\tCC(=O)OC1=CC=CC=C1C(=O)O\n"
        "4091\tCN(C)C(=N)N=C(N)N\n"
        "5743\tCN1CCC23C4OC5=C(O)C=CC(CC1C2C=CC4O)=C35\n"
    )
    _gz("CID-IUPAC.gz",
        "2244\t2-acetyloxybenzoic acid\n"
        "4091\t3-(diaminomethylidene)-1,1-dimethylguanidine\n"
    )
    _gz("CID-InChI-Key.gz",
        "2244\tBSYNRYMUTXBXSQ-UHFFFAOYSA-N\n"
        "4091\tXZWYZXLIPXDOLR-UHFFFAOYSA-N\n"
    )
    # CID-Mass.gz: CID<TAB>Formula<TAB>ExactMass<TAB>MonoIsotopicMass
    _gz("CID-Mass.gz",
        "2244\tC9H8O4\t180.04225873\t180.04225873\n"
        "4091\tC4H11N5\t129.098745\t129.098745\n"
        "5743\tC17H19NO3\t285.136493\t285.136493\n"
    )
    _gz("CID-PMID.gz",
        "2244\t11111\t1\n"
        "2244\t22222\t1\n"
        "4091\t22222\t1\n"
        "4091\t33333\t1\n"
        "9999\t44444\t1\n"  # not in target set
    )

    return bulk


def test_load_compound_index(tmp_path):
    bulk = _make_bulk_fixture(tmp_path)
    compounds = load_compound_index(bulk, require_mesh=True)

    assert len(compounds) == 3
    assert 2244 in compounds
    assert 4091 in compounds
    assert 5743 in compounds
    assert 9999 not in compounds  # not in CID-MeSH

    aspirin = compounds[2244]
    assert aspirin.name == "Aspirin"
    assert aspirin.smiles == "CC(=O)OC1=CC=CC=C1C(=O)O"
    assert aspirin.iupac_name == "2-acetyloxybenzoic acid"
    assert aspirin.inchi_key == "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
    assert aspirin.molecular_formula == "C9H8O4"
    assert aspirin.molecular_weight == 180.04225873
    assert "Aspirin" in aspirin.mesh_terms


def test_build_pmid_to_cids(tmp_path):
    bulk = _make_bulk_fixture(tmp_path)
    target_cids = {2244, 4091, 5743}
    pmid_to_cids, total = build_pmid_to_cids(bulk, target_cids)

    assert 11111 in pmid_to_cids
    assert pmid_to_cids[11111] == [2244]
    assert sorted(pmid_to_cids[22222]) == [2244, 4091]
    assert pmid_to_cids[33333] == [4091]
    assert 44444 not in pmid_to_cids  # CID 9999 not in target set


# ------------------------------------------------------------------
# PubMed XML parser
# ------------------------------------------------------------------


SAMPLE_PUBMED_XML = """<?xml version="1.0" encoding="utf-8"?>
<PubmedArticleSet>
<PubmedArticle>
  <MedlineCitation>
    <PMID Version="1">22222</PMID>
    <Article>
      <Journal>
        <Title>Journal of Pharmacology</Title>
        <JournalIssue>
          <PubDate><Year>2024</Year><Month>Mar</Month><Day>15</Day></PubDate>
        </JournalIssue>
      </Journal>
      <ArticleTitle>Aspirin and metformin combination therapy</ArticleTitle>
      <Abstract>
        <AbstractText>Aspirin inhibits COX-1 while metformin activates AMPK.</AbstractText>
      </Abstract>
      <AuthorList>
        <Author><LastName>Smith</LastName><ForeName>John</ForeName></Author>
        <Author><LastName>Doe</LastName><ForeName>Jane</ForeName></Author>
      </AuthorList>
      <Language>eng</Language>
      <PublicationTypeList>
        <PublicationType>Journal Article</PublicationType>
      </PublicationTypeList>
    </Article>
    <ChemicalList>
      <Chemical><NameOfSubstance>Aspirin</NameOfSubstance></Chemical>
      <Chemical><NameOfSubstance>Metformin</NameOfSubstance></Chemical>
    </ChemicalList>
    <MeshHeadingList>
      <MeshHeading>
        <DescriptorName MajorTopicYN="Y">Aspirin</DescriptorName>
        <QualifierName MajorTopicYN="Y">pharmacology</QualifierName>
      </MeshHeading>
      <MeshHeading>
        <DescriptorName>Metformin</DescriptorName>
        <QualifierName>therapeutic use</QualifierName>
      </MeshHeading>
      <MeshHeading>
        <DescriptorName>Humans</DescriptorName>
      </MeshHeading>
    </MeshHeadingList>
    <KeywordList>
      <Keyword>aspirin</Keyword>
      <Keyword>diabetes</Keyword>
    </KeywordList>
  </MedlineCitation>
  <PubmedData>
    <ArticleIdList>
      <ArticleId IdType="pubmed">22222</ArticleId>
      <ArticleId IdType="doi">10.1234/test.2024</ArticleId>
      <ArticleId IdType="pmc">PMC9876543</ArticleId>
    </ArticleIdList>
  </PubmedData>
</PubmedArticle>
<PubmedArticle>
  <MedlineCitation>
    <PMID Version="1">33333</PMID>
    <Article>
      <Journal><Title>Other Journal</Title></Journal>
      <ArticleTitle>Unrelated study</ArticleTitle>
      <Abstract><AbstractText>About something else.</AbstractText></Abstract>
    </Article>
  </MedlineCitation>
</PubmedArticle>
</PubmedArticleSet>
"""


def test_iter_pubmed_articles(tmp_path):
    xml_path = tmp_path / "test.xml.gz"
    with gzip.open(xml_path, "wt") as f:
        f.write(SAMPLE_PUBMED_XML)

    articles = list(iter_pubmed_articles(xml_path))
    assert len(articles) == 2

    a = articles[0]
    assert a.pmid == 22222
    assert a.title == "Aspirin and metformin combination therapy"
    assert "COX-1" in a.abstract
    assert len(a.authors) == 2
    assert a.authors[0] == "John Smith"
    assert a.journal == "Journal of Pharmacology"
    assert a.publication_date == "2024-03-15"
    assert a.doi == "10.1234/test.2024"
    assert a.pmcid == "PMC9876543"
    assert "Aspirin" in a.chemicals
    assert "Metformin" in a.chemicals
    assert a.language == "eng"
    assert "Journal Article" in a.publication_types

    # MeSH headings should preserve subheading format
    assert "*Aspirin/*pharmacology" in a.mesh_headings
    assert "Metformin/therapeutic use" in a.mesh_headings
    assert "Humans" in a.mesh_headings


def test_iter_pubmed_articles_with_filter(tmp_path):
    xml_path = tmp_path / "test.xml.gz"
    with gzip.open(xml_path, "wt") as f:
        f.write(SAMPLE_PUBMED_XML)

    # Only keep PMID 22222
    articles = list(iter_pubmed_articles(xml_path, target_pmids={22222}))
    assert len(articles) == 1
    assert articles[0].pmid == 22222


# ------------------------------------------------------------------
# End-to-end builder
# ------------------------------------------------------------------


def test_build_dataset_end_to_end(tmp_path):
    from chem2textqa.processing.builder import build_dataset

    bulk = _make_bulk_fixture(tmp_path)
    pubmed_dir = bulk / "pubmed_baseline"
    pubmed_dir.mkdir()

    with gzip.open(pubmed_dir / "pubmed_test.xml.gz", "wt") as f:
        f.write(SAMPLE_PUBMED_XML)

    output = tmp_path / "output.jsonl"
    stats = build_dataset(
        bulk_dir=bulk,
        output_path=output,
        require_abstract=True,
        require_qa_category=True,
    )

    assert stats.compounds_loaded == 3
    assert stats.articles_written >= 1

    # Read back the output
    records = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    pmids = {r["pmid"] for r in records}
    assert 22222 in pmids  # has both aspirin and metformin

    aspirin_record = next(r for r in records if r["pmid"] == 22222)
    assert "mechanism_of_action" in aspirin_record["qa_categories"]
    assert "therapeutic_use" in aspirin_record["qa_categories"]
    assert len(aspirin_record["linked_compounds"]) == 2  # 2244 + 4091
    cids = {c["cid"] for c in aspirin_record["linked_compounds"]}
    assert 2244 in cids
    assert 4091 in cids
