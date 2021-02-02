import time
import numpy as np

from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import create_parametric_leaf
from spn.algorithms.StructureLearning import learn_structure, get_next_operation
from spn.algorithms.splitting.Base import split_data_by_clusters

from simple_spn import functions as fn
from simple_spn import io_helper

'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def __get_split_skip():
    def split_skip(local_data, ds_context, scope):
        return split_data_by_clusters(local_data, clusters=np.array([1] * local_data.shape[1]), scope=scope, rows=False)

    return split_skip


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def _get_splitting_functions(cols, rows, ohe, threshold, rand_gen, n_jobs):
    from spn.algorithms.splitting.Clustering import get_split_rows_KMeans, get_split_rows_TSNE, get_split_rows_GMM
    from spn.algorithms.splitting.PoissonStabilityTest import get_split_cols_poisson_py
    from spn.algorithms.splitting.RDC import get_split_cols_RDC_py, get_split_rows_RDC_py

    if isinstance(cols, str):
        if cols == "rdc":
            split_cols = get_split_cols_RDC_py(threshold, rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
        elif cols == "poisson":
            split_cols = get_split_cols_poisson_py(threshold, n_jobs=n_jobs)
        elif cols == "None":
            split_cols = __get_split_skip()
        else:
            raise AssertionError("unknown columns splitting strategy type %s" % str(cols))
    else:
        split_cols = cols

    if isinstance(rows, str):
        if rows == "rdc":
            split_rows = get_split_rows_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=n_jobs)
        elif rows == "kmeans":
            split_rows = get_split_rows_KMeans()
        elif rows == "tsne":
            split_rows = get_split_rows_TSNE()
        elif rows == "gmm":
            split_rows = get_split_rows_GMM()
        else:
            raise AssertionError("unknown rows splitting strategy type %s" % str(rows))
    else:
        split_rows = rows
    return split_cols, split_rows


def _learn_parametric(
        data,
        ds_context,
        cols="rdc",
        rows="kmeans",
        min_instances_slice=200,
        min_features_slice=1,
        multivariate_leaf=False,
        cluster_univariate=False,
        threshold=0.3,
        ohe=False,
        leaves=None,
        memory=None,
        rand_gen=None,
        cpus=-1,
):
    if leaves is None:
        leaves = create_parametric_leaf

    if rand_gen is None:
        rand_gen = np.random.RandomState(17)

    def learn_param(data, ds_context, cols, rows, min_instances_slice, threshold, ohe):
        split_cols, split_rows = _get_splitting_functions(cols, rows, ohe, threshold, rand_gen, cpus)

        nextop = get_next_operation(min_instances_slice, min_features_slice, multivariate_leaf, cluster_univariate=cluster_univariate)

        return learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)

    if memory:
        learn_param = memory.cache(learn_param)

    return learn_param(data, ds_context, cols, rows, min_instances_slice, threshold, ohe)

'''
def create_parametric_leaf(data, ds_context, scope):
    from spn.structure.leaves.parametric.MLE import update_parametric_parameters_mle
    from spn.structure.leaves.parametric.Parametric import Categorical

    # assert len(scope) == 1, "scope of univariate parametric for more than one variable?"
    # assert data.shape[1] == 1, "data has more than one feature?"

    idx = scope[0]

    assert (
        ds_context.parametric_types is not None
    ), "for parametric leaves, the ds_context.parametric_types can't be None"
    assert (
        len(ds_context.parametric_types) > idx
    ), "for parametric leaves, the ds_context.parametric_types must have a parametric type at pos %s " % (idx)

    parametric_type = ds_context.parametric_types[idx]

    assert parametric_type is not None

    node = parametric_type()
    if parametric_type == Categorical:
        k = int(np.max(ds_context.domains[idx]) + 1)
        node = Categorical(p=(np.ones(k) / k).tolist())

    node.scope.extend(scope)

    update_parametric_parameters_mle(node, data)

    if parametric_type == Categorical:
        for i in range(len(node.p)):
            node.p[i] = (node.p[i] + 0.1) / (1.0 + len(node.p) * 0.1)

        print(str(sum(node.p)) + " " + str(node.p))


    return node
'''


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def get_standard_parametric_types(feature_types):
    return fn.get_standard_parametric_types(feature_types)


def learn_parametric_spn(data, parametric_types, rdc_threshold=0.3, min_instances_slice=0.05, colSplit="rdc",
                         rowSplit="kmeans", cluster_univariate=False):
    ds_context = Context(parametric_types=parametric_types)
    ds_context.add_domains(data)
    mis = int(len(data) * min_instances_slice)

    t0 = time.time()
    spn = _learn_parametric(data, ds_context, threshold=rdc_threshold, min_instances_slice=mis, cols=colSplit,
                            rows=rowSplit, cluster_univariate=cluster_univariate)
    const_time = time.time() - t0

    return spn, const_time


def learn_parametric_spns(dataset_name, rdc_thresholds=[0.3], min_instances_slices=[0.05], colSplits=["rdc"],
                          rowSplits=["kmeans"], cluster_univariates=[False]):
    trans_df, f_info, parametric_types = io_helper.load(dataset_name, loc="_cache/spn")

    for rdc_threshold in rdc_thresholds:
        for min_instances_slice in min_instances_slices:
            for colSplit in colSplits:
                for rowSplit in rowSplits:
                    for cluster_univariate in cluster_univariates:
                        spn, const_time = learn_parametric_spn(trans_df.values, parametric_types,
                                                               rdc_threshold=rdc_threshold,
                                                               min_instances_slice=min_instances_slice, colSplit=colSplit,
                                                               rowSplit=rowSplit, cluster_univariate=cluster_univariate)
                        save_spn(spn, const_time, dataset_name, rdc_threshold, min_instances_slice, colSplit=colSplit,
                                 rowSplit=rowSplit, cluster_univariate=cluster_univariate, f_info=f_info)


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def get_spn_ident(rdc_threshold, min_instances_slice, colSplit="rdc", rowSplit="kmeans", cluster_univariate=False):

    if cluster_univariate:
        return "colSplit=" + str(colSplit) + "_rowSplit=" + str(rowSplit) + "_rdc=" + str(rdc_threshold) + "_mis=" + str(min_instances_slice) + "_cluster_univariate=" + str(cluster_univariate)
    else:
        return "colSplit=" + str(colSplit) + "_rowSplit=" + str(rowSplit) + "_rdc=" + str(rdc_threshold) + "_mis=" + str(min_instances_slice)


def get_spn(dataset_name, rdc_threshold, min_instances_slice, colSplit="rdc", rowSplit="kmeans", cluster_univariate=False):
    if not exist_spn(dataset_name, rdc_threshold, min_instances_slice, colSplit, rowSplit):
        learn_parametric_spns(dataset_name, rdc_thresholds=[rdc_threshold], min_instances_slices=[min_instances_slice],
                              colSplits=[colSplit], rowSplits=[rowSplit], cluster_univariate=[cluster_univariate])
    return load_spn(dataset_name, rdc_threshold, min_instances_slice, colSplit, rowSplit, cluster_univariate=cluster_univariate)


def exist_spn(dataset_name, rdc_threshold, min_instances_slice, colSplit="rdc", rowSplit="kmeans", cluster_univariate=False):
    return io_helper.exist(get_spn_ident(rdc_threshold, min_instances_slice, colSplit, rowSplit, cluster_univariate=cluster_univariate), dataset_name, "_spns")


def save_spn(spn, const_time, dataset_name, rdc_threshold, min_instances_slice, colSplit="rdc", rowSplit="kmeans", cluster_univariate=False,
             f_info=None):
    if f_info is None: fn.generate_adhoc_feature_info(spn)
    io_helper.save([spn, f_info, const_time], get_spn_ident(rdc_threshold, min_instances_slice, colSplit, rowSplit, cluster_univariate=cluster_univariate), loc="_spns/" + dataset_name)


def load_spn(dataset_name, rdc_threshold, min_instances_slice, colSplit="rdc", rowSplit="kmeans", cluster_univariate=False):
    return io_helper.load(get_spn_ident(rdc_threshold, min_instances_slice, colSplit, rowSplit, cluster_univariate=cluster_univariate), loc="_spns/" + dataset_name)


'''
***********************************************************************************************************
***********************************************************************************************************
***********************************************************************************************************
'''


def create_spn_data(df, dataset_name, feature_types=None):
    spn_data = fn.transform_dataset(df, feature_types)
    io_helper.save(spn_data, dataset_name, loc="_cache/spn_data")


def save_spn_data(dataset_name, df, f_info, feature_types):
    io_helper.save([df, f_info, feature_types], dataset_name, loc="_cache/spn_data")


def exist_spn_data(dataset_name):
    return io_helper.exist(dataset_name, loc="_cache/spn_data")


def load_spn_data(dataset_name):
    return io_helper.load(dataset_name, loc="_cache/spn_data")

