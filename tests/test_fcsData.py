import pytest, os
from src.fcsdata_object import fcsData, FailedDistanceCalculationError, ClusterMemberError
from collections import Counter


@pytest.fixture()
def gvhd_test_object():
    test_path = os.path.dirname(os.path.abspath(__file__))
    test_obj = fcsData(filename="{}/fcs_data/gvhd_001.fcs".format(test_path))
    return test_obj

@pytest.fixture()
def embg_test_object():
    test_path = os.path.dirname(os.path.abspath(__file__))
    test_obj = fcsData(filename="{}/fcs_data/Modern_Betula.fcs".format(test_path))
    return test_obj


def test_init(gvhd_test_object):
    assert gvhd_test_object.parameters == {"FSC.H":False, "SSC.H":False, "FL1.H":False,
                                      "FL2.H":False,"FL3.H":False,"FL4.H":False}
    assert gvhd_test_object.clustering is None
    assert gvhd_test_object._cluster_centroids is None
    assert gvhd_test_object._brushed is False
    assert gvhd_test_object.discarded is False
    assert gvhd_test_object.discarded_count == 0
    assert gvhd_test_object._selection is None


def test_set_parameters_for_analysis(gvhd_test_object):
    # case 01: valid input
    test_dict = {"FSC.H":True, "FL3.H":True}
    gvhd_test_object.setParametersForAnalysis(test_dict)
    assert gvhd_test_object.parameters == {"FSC.H": True, "SSC.H": False, "FL1.H": False,
                                      "FL2.H": False, "FL3.H": True, "FL4.H": False}
    # case 02: invalid input
    test_dict = {"ABC" : True}
    assert gvhd_test_object.setParametersForAnalysis(test_dict) is False


def test_to_csv(gvhd_test_object, tmpdir):
    gvhd_test_object.toCSV(filename=tmpdir.join("test.csv"))
    assert len(tmpdir.listdir()) == 1

def test_apply_selection(gvhd_test_object):
    # case 01: a subset of the data is selected
    selection = gvhd_test_object._dataset.iloc[0:100]
    gvhd_test_object.applySelection(selection)
    assert gvhd_test_object._brushed is True
    assert selection.equals(gvhd_test_object._selection)
    # case 02: the entire data set is selected
    selection = gvhd_test_object._dataset.iloc[:]
    gvhd_test_object.applySelection(selection)
    assert gvhd_test_object._brushed is False


def test_return_selected_dimensions(gvhd_test_object):
    param = {"FSC.H": True, "SSC.H": True, "FL1.H": False,
             "FL2.H": False, "FL3.H": True, "FL4.H": False}
    gvhd_test_object.setParametersForAnalysis(param)
    selected_data = gvhd_test_object.returnSelectedDimensions()
    assert isinstance(selected_data, dict) # check return type
    # and check if keys in dictionary entries corresponding to data points only contains the selected dimensions
    assert Counter(list(selected_data[0].keys())) == Counter(["FSC.H", "SSC.H", "FL3.H"])


def test_return_selected_dimensions_labels(gvhd_test_object):
    param = {"FSC.H": True, "SSC.H": True, "FL1.H": False,
             "FL2.H": True, "FL3.H": False, "FL4.H": False}
    gvhd_test_object.setParametersForAnalysis(param)
    assert Counter(gvhd_test_object.returnSelectedDimensionsLabels()) == Counter(["FSC.H", "SSC.H", "FL2.H"])


def test_get_centroids_for_plot(gvhd_test_object):
    # case 01: no clustering has been performed
    assert gvhd_test_object.getCentroidsForPlot() is False
    # case 02: run clustering and determine if correct number of clusters is returned
    param = {"FSC.H":True,"SSC.H":True,"FL1.H":True,
             "FL2.H":True,"FL3.H":True,"FL4.H":True}
    gvhd_test_object.setParametersForAnalysis(param)
    gvhd_test_object.flowMeans(k=20, selection=True, discard_small_clusters=False)
    assert isinstance(gvhd_test_object.getCentroidsForPlot(), dict)
    # as many centroids as clusters should be returned
    assert len(gvhd_test_object.getCentroidsForPlot().keys()) == gvhd_test_object.clust_no
    # and check that the keys for centroids are the same as set for analysis
    assert Counter(list(gvhd_test_object.getCentroidsForPlot()[0].keys())) == \
           Counter(list(param.keys()))


def test_get_cluster_for_plot(gvhd_test_object):
    # case 01: no clustering has been performed
    assert gvhd_test_object.getClusterForPlot(index=0) is None
    # case 02: clustering performed, check that all relevant elements are there
    param = {"FSC.H": True, "SSC.H": True, "FL1.H": True,
             "FL2.H": True, "FL3.H": True, "FL4.H": True}
    gvhd_test_object.setParametersForAnalysis(param)
    gvhd_test_object.flowMeans(k=20, selection=True, discard_small_clusters=False)
    for i in range(gvhd_test_object.clust_no):
        cluster = gvhd_test_object.getClusterForPlot(index=i)
        assert isinstance(cluster, dict)
        assert len(cluster) == 3
        assert Counter(list(cluster[0].keys())) == Counter(param)
        assert Counter(list(cluster[1].keys())) == Counter(param)
        assert Counter(list(cluster[2].keys())) == Counter(param)


def test_get_general_stats(gvhd_test_object):
    stats = gvhd_test_object.getGeneralStats()
    assert isinstance(stats, dict)
    assert stats["Events"] == gvhd_test_object._dataset.shape[0]
    for par in ["FSC.H", "SSC.H", "FL1.H", "FL2.H", "FL3.H", "FL4.H"]:
        assert par in stats["Channels"]


def test_get_selection_stats(gvhd_test_object):
    # case 01: no selection has been applied
    assert gvhd_test_object.getSelectionStats()["Brushed"] is False
    # case 02: a selection has been applied
    param = ["FSC.H", "SSC.H", "FL1.H", "FL2.H", "FL4.H"]
    selection = gvhd_test_object._dataset[param].iloc[:100]
    gvhd_test_object.applySelection(selection)
    stats = gvhd_test_object.getSelectionStats()
    assert stats.pop("Brushed") is True
    assert stats.pop("Events") == 100
    assert Counter(list(stats.keys())) == Counter(["FSC.H", "SSC.H", "FL1.H", "FL2.H", "FL4.H"])

def test_get_cluster_stats(gvhd_test_object):
    # case 01: no clustering completed
    assert gvhd_test_object.getClusterStats() is None

    # case 02: clustering completed
    param = {"FSC.H": True, "SSC.H": True, "FL1.H": True,
             "FL2.H": True, "FL3.H": True, "FL4.H": True}
    gvhd_test_object.setParametersForAnalysis(param)
    gvhd_test_object.flowMeans(k=20, selection=True, discard_small_clusters=False)
    stats = gvhd_test_object.getClusterStats()
    # stats is a dictionary with n entries (where n is number of clusters)
    assert Counter(list(stats.keys())) == Counter([str(x) for x in range(1, gvhd_test_object.clust_no + 1)])
    key_list = list(param.keys())
    key_list.append("Events")
    for key in stats.keys():
        # every cluster entry contains data for all dimensions
        assert Counter(key_list) == Counter(list(stats[key].keys()))
        stats[key].pop("Events")
        for key, item in stats[key].items():
            assert hasattr(item, "min")
            assert hasattr(item, "max")
            assert hasattr(item, "mean")
            assert hasattr(item, "std")


def test_flowMeans(gvhd_test_object, embg_test_object):
    # case 01: successful clustering should be possible without discarding outliers
    param = {"FSC.H": True, "SSC.H": True, "FL1.H": True,
             "FL2.H": True, "FL3.H": True, "FL4.H": True}
    gvhd_test_object.setParametersForAnalysis(param)
    gvhd_test_object.flowMeans(k=20, selection=True, discard_small_clusters=False)
    assert gvhd_test_object.clustering is not None
    assert gvhd_test_object._cluster_centroids is not None
    # case 02: no clustering possible without discarding
    param = {"FSC-A":True, "SSC-A":True, "B 530/30-A":True, "B 585/42-A":True,
             "B 616/23 NR-A":True, "B 695/40-A":True, "B 780/60-A":True,
             "R 660/20-A":True, "R 780/60-A":True, "V 450/40-A":True, "V 530/30-A":True}
    embg_test_object.setParametersForAnalysis(param)
    with pytest.raises(ClusterMemberError) as e_info:
        embg_test_object.flowMeans(k=20, selection=True, discard_small_clusters=False)
    assert embg_test_object.clustering is None
    assert embg_test_object._cluster_centroids is None
    # case 03: clustering only possible with discarding
    assert embg_test_object.flowMeans(k=20, selection=True, discard_small_clusters=True)
    assert embg_test_object.clustering is not None
    assert embg_test_object._cluster_centroids is not None
