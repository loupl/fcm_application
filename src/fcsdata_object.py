import fcsparser
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.cluster import KMeans
from sklearn import preprocessing
from scipy.spatial.distance import mahalanobis

# hidden imports -> required by PyInstaller to create appropriate hooks
import pandas._libs.tslibs.timedeltas
import scipy._lib.messagestream
import sklearn.neighbors.typedefs
import sklearn.neighbors.quad_tree
import sklearn.tree
import sklearn.tree._utils


class ClusterMemberError(Exception):
    """
    Simple custom exception to be raised when a cluster determined by k-Means contains fewer data points than
    the data points have dimensions.
    In this case further processing is not possible because the covariance matrix of this cluster isnot invertible.
    This case also suggests that the members of this cluster is most likely an outlier that should have been
    discarded during pre-processing.
    """
    pass

class FailedDistanceCalculationError(Exception):
    """
    Simple custom exception to be raised when a distance calculation in the flowMeans algorithm fails. This is likely
    to occur when the covariance matrix of one of the clusters is not invertible.
    """
    pass

class fcsData:
    """
    An object holding flow cytometry data as read from a .fcs file (FCS 2.0, 3.0, 3.1 as determined by
    fcsparser package). It allows the following operations:
        - writing the raw data to CSV
        - retrieval of metadata, raw data, cluster centroids and statistical information about selection and cluster
        - manipulation using the flowMeans clustering algorithms (Aghaeepour et al, 2011)
        - storage of manipulation results (e.g. from manual selection in visualisation)
        -
    """

    def __init__(self, filename):
        """
        Instantiate a fcsData instance. Variable indicating if dimension should be used for analysis set to False
        for all dimensions by default. Further variables containing details about selection or analysis are set
        to 0 or None by default.

        :param filename: path to the FCS file to be read; instance of this class is instantiated with the data
        read from this file as read by fcsparser package (https://github.com/eyurtsev/fcsparser)
        """
        self._metadata, self._dataset = fcsparser.parse(path=filename, meta_data_only=False, reformat_meta=True)
        self._channels = self._metadata['_channels_']
        self.parameters = {} # stores the different parameters contained in the .fcs file and whether they should be
                             # be included in the analysis - visualisation, clustering (True if yes, False if no)
        for channel in self._channels['$PnN']: # False as default for all parameters
            self.parameters[channel] = False
        # initialise variables holding the clustering information
        self._clustering = None
        self._cluster_centroids = None
        self._brushed = False
        self._discarded = False
        self._discarded_count = 0
        self._selection = None # store selection (e.g from brush)

    def _get_channels(self):
        return self._channels

    def _get_metadata(self):
        return self._metadata

    def _get_dataset(self):
        return self._dataset

    def _get_clustering(self):
        return self._clustering

    def _get_discarded(self):
        return self._discarded

    def _set_discarded(self, val):
        self._discarded = val

    def _get_discarded_count(self):
        return self._discarded_count

    def _get_clust_no(self):
        if self._clustering is not None:
            clust_ids = np.unique(self._clustering)
            return len(clust_ids)

    # define properties once getters/setters have been defined
    channels = property(_get_channels)
    clustering = property(_get_clustering)
    discarded = property(_get_discarded, _set_discarded)
    discarded_count = property(_get_discarded_count)
    clust_no = property(_get_clust_no)

    # ----------------------------------- methods for internal use -----------------------------------------------------
    def setParametersForAnalysis(self, parameters):
        """
        Sets boolean values for all channels contained in the data according to choice of user
            -> True: parameter should be included in analysis
            -> False: parameter should be excluded from analysis
        These settings will affect what is written to the JSON file, which forms the basis of the parallel coordinate
        plots displayed in the application.
        :param parameters: a dictionary containing the selection of the user; keys should be channel names as contained
                           in the file being analysed
        :return: True if choice could be successfully set for all parameters; False in case any inconsistencies arising
                 from names in parameters argument and parameters dictionary associated with object occur
        """
        for par, val in parameters.items():
            if par in self.parameters:
                self.parameters[par] = val
            else:
                return False  # if parameter chosen is not in parameters based on channels read from file
                # there is some inconsistency in the data being passed (most likely name of channels)
        return True  # if all parameters can be successfully iterated over, there should be no inconsistencies and
                     # parameters are set

    def toCSV(self, filename):
        """
        Write the raw data (i.e. all numerical values of the data points' dimensions) to CSV. A header with the
        dimension names is included. In case outliers have been discarded as describe in flowMeans, these will not
        be included in the CSV file.

        :param filename: name (incl. path) of the file to which to write
        """
        self._dataset.to_csv(path_or_buf=filename, header=True, index=False, columns=list(self.parameters.keys()))

    def applySelection(self, selection):
        """
        Set a subset of the data as the selected subset. Information about this can be retrieved using other
        methods of the fcsData class. In case all data points are contained in the selection, this instance variable
        is set to None indicating that no selection was made.

        :param selection: Pandas Dataframe containing the selection.
        """
        if selection.shape[0] == self._dataset.shape[0]:
            # if all data is in the brush / brush has been removed
            self._selection = None
            self._brushed = False
        else:
            self._selection = selection
            self._brushed = True

    def returnSelectedDimensions(self):
        """
        Returns a dictionary containing all events stored in the fcsData object for all dimensions (parameters) selected
        by the user through setParametersForAnalysis method.

        :return: A dictionary for which each key is the index of the corresponding data point (event) and the value is a
                 dictionary containing key-value pairs of the selected dimensions in the form (name : value).
        """
        columns = []
        for par in self.parameters:
            if self.parameters[par]:
                columns.append(par)
        data = self._dataset[columns]
        data = data.to_dict('index')
        return data

    def returnSelectedDimensionsLabels(self):
        """
        :return: list with the names of all dimensions selected
        """
        columns = list()
        for par in self.parameters:
            if self.parameters[par] == True:
                columns.append(par)
        return columns

    def getCentroidsForPlot(self):
        """
        :return: Cluster centroids as a Panda DataFrame. In case no clustering information is avilable, False
        will be returned.
        """
        if self._cluster_centroids is not None:
            columns = []
            for par in self.parameters:
                if self.parameters[par]:
                    columns.append(par)
            centroids = pd.DataFrame(data=self._cluster_centroids,
                                     columns=columns)
            centroids = centroids.to_dict('index')
            return centroids
        else:
            return False

    def getClusterForPlot(self, index):
        """
        :param index: numerical index of the cluster about which information is returned; index is between 0 and
         k-1 for a clustering solution with k clusters
        :return: details about the selected cluster as a dictionary; keys correspond to dimension labels; value
        at index 0 is minimum, index 1 is mean, index 2 is maximum
        """
        if self._clustering is not None:
            indices = np.where(self._clustering == index)[0]
            cluster = self._dataset.iloc[indices]
            columns = []
            for par in self.parameters:
                if self.parameters[par]:
                    columns.append(par)
            min = cluster[columns].min().values
            mean = cluster[columns].mean().values
            max = cluster[columns].max().values
            clust = pd.DataFrame(columns=columns)
            clust.loc[0] = min
            clust.loc[1] = mean
            clust.loc[2] = max
            return clust.to_dict('index')

    def getGeneralStats(self):
        """
        Return information about the data set stored in an instance of fcsData.
            Events : number of events in data set
            Channels : name of each dimension in the data set (formatted for HTML display with <br>
            between each name)
        :return: dictionary containing details about data set in fcsData instance as described above
        """
        stats = dict()
        stats["Events"] = self._dataset.shape[0]
        channels = ""
        c_no = self._channels["$PnN"].shape[0]
        c_count = 1
        for c in self._channels["$PnN"]:
            # don't add a break after the last channel
            if c_count < c_no:
                channel = "{}<br>".format(str(c))
            else:
                channel = str(c)
            c_count += 1
            channels += channel
        stats["Channels"] = channels
        return stats

    def getSelectionStats(self):
        """
        Return information about the selection applied to an instance of fcsData, in case this is available.
        Details returned are:
            Brushed : indicates if selection is applied (Boolean)
            Events : number of events in selection
            For each dimension name : named tuple with min, max, mean attribute containing the respective
            values for this dimension
        :return: dictionary with details as described above
        """
        Dimension = namedtuple('Dimension', 'min max mean')
        stats = dict()
        stats["Brushed"] = self._brushed
        if self._brushed:
            stats["Events"] = self._selection.shape[0]
            # get the stats for each dimension
            for column in self._selection:
                stats[column] = Dimension(min=self._selection[column].min(),
                                          max=self._selection[column].max(),
                                          mean=self._selection[column].mean())
        return stats

    def getClusterStats(self):
        """
        Return information about the clustering solution determined by flowMeans, should this be available.
        The details included are the following for each cluster, which can be accessed using its index as a key:
            Events : number of events (data points) in this cluster
            For each dimension : named tuple containing min, max, mean, standar deviation (std) attribute with
            the corresponding values
        :return: dictionary containing the details as described above.
        """
        Dimension = namedtuple('Dimension', 'min max mean std')
        stats = dict()
        if self._clustering is None:
            return
        else:
            columns = list()
            for par in self.parameters:
                if self.parameters[par]:
                    columns.append(par)
            data = self._dataset[columns]

            clust_ids = np.unique(self._clustering)
            all_indices = list()
            for i in clust_ids:
                # get the indices for each cluster
                indices = np.where(self._clustering == i)[0].tolist()
                all_indices.append(indices)

            for count, ind in enumerate(all_indices, 1):
                clust = data.loc[ind]
                clust_stats = dict()
                clust_stats["Events"] = clust.shape[0]
                for column in clust:
                    clust_stats[column] = Dimension(min=clust[column].min(),
                                                    max=clust[column].max(),
                                                    mean=clust[column].mean(),
                                                    std=clust[column].std())
                stats[str(count)] = clust_stats

        return stats

    # ---------------------------------- methods for manipulating data -------------------------------------------------
    def flowMeans(self, k, selection=True, discard_small_clusters=False):
        """
        An extensive description of the original version of this  algorithm is available in:
             N. Aghaeepour, R. Nikolic, H. H. Hoos, and R. R. Brinkman. Rapid Cell Population Identification in Flow Cytometry Data. Cytometry Part A, 79(1), 2011

        The algorithm was adapted to discard outliers automatically should this be desired. Outliers are defined
        as all points in clusters which contain fewer members than dimensions included in the analysis.

        The clustering algorithm determines a clustering solution based on the following steps:
            1. determine initial clustering using k-means (scikit-learn implementation) using parameter k
            1.1 discard outliers and determine new clustering solution with standard k-means as long as outliers exist
            2. determine breakpoint using segmented regression
            3. merge clusters up to breakpoint
        A more extensive description is provided in Aghaeepour et al., 2011.

        The method throws ClusterMemberError, FailedDistanceCalculationError and passes on any other error
        raised by numpy, pandas or scikit-learn.

        :param k: input for the standard k-means used as part of flowMeans
        :param selection: variable indicating if only the parameters selected to be included in the analysis should
        be used during clustering or if all available dimensions should be used
        :param discard_small_clusters: variable indicating if small clusters, i.e. outliers, should be discarded
        automaatically
        :return: True in case clustering solution could be determined; in this case the instance variables
        self._clustering, self._cluster_centroids, self._discarded = False, self._discarded_count have been set
        accordingly
        """

        # define inner function for use in segmented regression problem
        def lin_reg(weights, v):
            return weights[0] + weights[1] * v

        # define inner function for merging nearest clusters
        def merge_nearest(data, clustering):
            """

            :param data:
            :param clustering:
            :return:
            """
            cluster_ids, cluster_lengths = np.unique(clustering, return_counts=True)  # determine unique cluster ids
            all_indeces, covariances, centroids = list(), list(), list()  # initialise

            for i in cluster_ids:
                # this information can be easily obtained for the initial merge step from the sk-learn model;
                # however, for the following merge steps this needs to be recorded separately
                indeces = np.where(clustering == i)
                all_indeces.append(indeces)  # store all indeces belonging to i-th cluster
                covariances.append(np.cov(data[indeces, :][0], rowvar=False))
                centroids.append(np.mean(data[indeces, :][0], axis=0))

            no_of_clusters = len(cluster_ids)
            pairwise_distances = np.zeros((no_of_clusters, no_of_clusters))
            for i in range(no_of_clusters):
                for j in range(no_of_clusters):
                    # record the Symmetric Mahalanobis Semi-Distances as described in Aghaeepour et al., 2011
                    if j > i:
                        # only record entries to the "north-east" of diagonal to avoid repetitions
                        try:  # use try-except as LinAlgError occurred during development and experimentation
                            # calculate the Mahalanobis distance based on centroids
                            dist_ij = mahalanobis(centroids[i], centroids[j], np.linalg.inv(covariances[i]))
                            dist_ji = mahalanobis(centroids[j], centroids[i], np.linalg.inv(covariances[j]))

                            # scipy mahalanobis() method will return NaN in case the inner term of the Mahalanobis
                            # distance is negative (which is caused by the covariance matrix not being positive,
                            # semi-definite) -> this is checked for below because working with NaN in the comparison
                            # of distances later in the algorithm results in these values always being recorded as the
                            # minimum distance, which is an obvious error causing problems further on when the
                            # segmented regression problem is solved
                            if np.isnan(dist_ji) or np.isnan(dist_ij):
                                # if one of the two values for the symmetric semi-distance is NaN, the calculation
                                # should fail
                                size_clust_i = all_indeces[i][0].shape[0]
                                size_clust_j = all_indeces[j][0].shape[0]
                                inner_term_ij = np.dot(np.dot((centroids[i] - centroids[j]),
                                                              np.linalg.inv(covariances[i])),
                                                       (centroids[i] - centroids[j]).T)
                                inner_term_ji = np.dot(np.dot((centroids[j] - centroids[i]),
                                                              np.linalg.inv(covariances[j])),
                                                       (centroids[j] - centroids[i]).T)
                                # html formatting for improved readibility in PyQt application
                                message = "An error occured during the calculation of the pairwise semi-distance " \
                                          "(Aghaeepour et al., 2011). <br> i: {} <br> j: {} <br>" \
                                          "size of i: {} <br> size of j: {} <br>" \
                                          "inner term ij: {} <br>" \
                                          "inner term ji: {}".format(i, j, size_clust_i, size_clust_j,
                                                                     inner_term_ij, inner_term_ji)
                                raise FailedDistanceCalculationError(message)
                        except np.linalg.linalg.LinAlgError:
                            # error appears to be caused by singularity in covariance matrix
                            # print('LinAlgError occurred')
                            # this can't be effectively addressed, therefore raise the error
                            raise

                        # use the smaller of the two distances
                        if dist_ij < dist_ji:
                            pairwise_distances[i, j] = dist_ij
                        else:
                            pairwise_distances[i, j] = dist_ji

            min_dist = np.min(
                pairwise_distances[np.nonzero(pairwise_distances)])  # determine the smallest distance overall
            min_dist_ind = np.where(pairwise_distances == min_dist)  # retrieve the corresponding indices
            min_dist_ind = (min_dist_ind[0][0], min_dist_ind[1][0])
            cluster01_indeces = all_indeces[min_dist_ind[0]][0]
            cluster02_indeces = all_indeces[min_dist_ind[1]][0]
            merged_cluster = np.append(cluster01_indeces, cluster02_indeces, axis=0)  # merge the clusters' indices

            # record the new clustering
            new_clustering = np.copy(clustering)

            if min_dist_ind[0] < min_dist_ind[1]:
                # determine the smaller cluster id to "re-index" the clusters
                smaller_ind = min_dist_ind[0]
                bigger_ind = min_dist_ind[1]
            else:
                smaller_ind = min_dist_ind[1]
                bigger_ind = min_dist_ind[0]

            for i in range(new_clustering.shape[0]):
                # check if i-th element is in merged cluster
                if np.any(np.isin(merged_cluster, i)):
                    if clustering[i] in min_dist_ind:
                        # and reassign cluster id
                        new_clustering[i] = smaller_ind

            # avoid 'gaps' in the cluster ids by reducing all ids greater than the bigger of the two ids merged by one
            for i in range(new_clustering.shape[0]):
                if new_clustering[i] > bigger_ind:
                    new_clustering[i] -= 1

            return new_clustering, min_dist

        #  get the data to be clustered based on user's choice of how clustering should be done
        if selection:
            columns = []
            for par in self.parameters:
                if self.parameters[par] == True:
                    columns.append(par)
            data = self._dataset[columns].values.astype(float)
        else:
            columns = self.parameters # required to determine appropriate cluster size in step 1.1
            data = self._dataset.values.astype(float)

        data = preprocessing.scale(data)

        # step 1: standard k-means with euclidean distance
        model = KMeans(n_clusters=k).fit(data)  # run the sk-learn k-means to determine initial clustering
        clustering = model.predict(data)

        # step 1.1: catch clusters which too few members -> these are highly likely to be artifacts/outliers
        cluster_ids, cluster_lengths = np.unique(clustering, return_counts=True)
        clust_id_remove = np.where(cluster_lengths < len(columns))[0]  # returns indices which start at 0
        count_iter = 0
        while len(clust_id_remove) > 0: # if there are any clusters that are too small
            if discard_small_clusters:
                count_iter += 1
                indeces_to_remove = list()
                for id_remove in clust_id_remove:
                    ids = np.where(clustering == id_remove)[0]
                    indeces_to_remove.extend(ids)
                self._discarded_count += len(indeces_to_remove)
                self._dataset.drop(indeces_to_remove, inplace=True)
                self._dataset.reset_index(drop=True, inplace=True)
                self._discarded = True
                data = preprocessing.scale(self._dataset[columns].values.astype(float))
                model = KMeans(n_clusters=k).fit(data)  # run the sk-learn k-means to determine new clustering
                clustering = model.predict(data)
                cluster_ids, cluster_lengths = np.unique(clustering, return_counts=True)
                clust_id_remove = np.where(cluster_lengths < len(columns))[0]  # returns indices which start at 0
            else:
                message = '<br>{} cluster(s) with too few members (number of members less than {} ' \
                          'dimensions considered) has/have been found. <br>'.format(
                    len(clust_id_remove), len(columns))
                cluster_sizes = list()
                for id_remove in clust_id_remove:
                    cluster_sizes.append(np.where(clustering==id_remove)[0].shape[0])
                message += "The clusters contain {} data points respectively.".format(cluster_sizes)
                message += "<br><hr><br>"
                message += "This means that the covariance matrix of these clusters is not invertible. <br>" \
                           "Therefore, the Mahalanobis between any of these clusters and any other cluster cannot "\
                           "be calculated. <br> This is most likely caused by outliers being present in the data " \
                           "set. <br> Please remove these outliers manually using a different software or <br>" \
                           "run <i>flowMeans</i> again with the option to automatically discard these data points."\
                           "<br> For more details refer to the documentation."
                raise ClusterMemberError(message)

        # step 2: merge nearest clusters until only one remains and record distances for segmented regression
        min_distances = np.zeros((k - 1,))
        new_clustering = np.copy(clustering)
        for i in range(k - 1):
            new_clustering, min_dist = merge_nearest(data, new_clustering)
            min_distances[-(i + 1)] = min_dist

        # step 3: solve the segmented regression problem
        total_errors = list()  # record the total error of both regression problems to choose breakpoint

        for i in range(2, len(min_distances) - 1):
            # solve the individual regression problems by moving breakpoint from 2 to k-2
            bp = i
            # problem 01
            dm01 = np.ones((bp, 2))
            for j in range(bp):
                dm01[j, 1] = j + 1
            weights01 = np.linalg.pinv(dm01) @ min_distances[:bp]
            errors01 = np.zeros((bp,))
            for j in range(len(errors01)):
                errors01[j] = (lin_reg(weights01, j) - min_distances[j]) ** 2
            # problem 02
            dm02 = np.ones((len(min_distances) - bp, 2))
            for j in range(0, len(min_distances) - bp):
                dm02[j, 1] = j + 1 + bp
            weights02 = np.linalg.pinv(dm02) @ min_distances[bp:]
            errors02 = np.zeros((len(min_distances) - bp,))
            for j in range(len(errors02)):
                errors02[j] = (lin_reg(weights02, j + bp) - min_distances[j + bp]) ** 2

            total_errors.append((np.sum(errors01) + (np.sum(errors02))))

        optimum_bp = np.where(total_errors == min(total_errors))[0][0] + 2  # add two because starting with
        # i = 2 which corresponds to index of optimal breakpoint but list of total errors indexed from 0

        # step 4: merge up to the optimum breakpoint
        final_clustering = np.copy(clustering)
        for i in range(k - optimum_bp):
            # store second return value (min_distance) in placeholder
            final_clustering, _ = merge_nearest(data, final_clustering)

        # and determine the centroids for the final clusters
        final_ids = np.unique(final_clustering)
        final_centroids = np.zeros((final_ids.shape[0], data.shape[1]))
        for i in final_ids:
            indeces = np.where(final_clustering == i)
            final_centroids[i] = np.mean(self._dataset[columns].values[indeces, :], axis=1)

        self._clustering, self._cluster_centroids = final_clustering, final_centroids
        return True
