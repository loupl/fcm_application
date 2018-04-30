from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWebEngineWidgets import *
from flask import Flask, render_template
import socketio
import sys
import os
import numpy as np
import pandas as pd # required for fcs parser
# custom import
from src.fcsdata_object import fcsData, ClusterMemberError, FailedDistanceCalculationError


def resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller.
    """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


if getattr(sys, 'frozen', False):
    template_folder = resource_path('templates')
    static_folder = resource_path('static')
    flask_app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
else:
    flask_app = Flask(__name__)


class MainWindow(QMainWindow):
    """
    Main window of the Flow Cytometry application. It contains the menu bar for accessing the application's
    functionality and the QWebEngine to display the parallel coordinate visualisation. The data being analysed
    is stored as a fcsData instance.
    """

    updateBrushStats = pyqtSignal(dict)

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # bind dataset to window
        self.dataset = None
        self.filename = None
        self.stats_dlg = None # reference to this needs to be stored to use .show()
        self.pcp_displayed = False # state variable to keep track of use of "canvas" in webview
        self.threadpool = QThreadPool()
        self.wait_dlg = None # see above (stats_dlg)
        self.doc_dlg = DocumentationDialog()
        self.set_up()

    def set_up(self):
        """
        Method to initialise basic window properties (window title, menu bar, etc.). Connects menu bar fields,
        buttons with the corresponding functions.
        """
        # define initial state
        w, h = 1400, 1200
        self.setWindowTitle('Flow Cytometry Application')
        self.resize(w, h)

        layout = QVBoxLayout()

        web_view = QWebEngineView()
        web_view.setMaximumSize( w , h-300 )
        webview_layout = QHBoxLayout()
        webview_layout.addWidget(web_view)

        layout.addLayout(webview_layout)
        main_widget = QWidget()
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # set up menubar
        self.menuBar().setNativeMenuBar(True)
        file_menu = self.menuBar().addMenu('&File')
        data_menu = self.menuBar().addMenu('&Data')
        vis_menu = self.menuBar().addMenu('&Visualisation')
        about_menu = self.menuBar().addMenu('&Documentation')

        # add actions to menu bar
        open_file_action = QAction(QIcon(), 'Open file', self)
        open_file_action.triggered.connect(self.open_file)
        file_menu.addAction(open_file_action)
        self.save_to_csv_action = QAction(QIcon(), 'Save to .csv', self)
        self.save_to_csv_action.setDisabled(True)
        self.save_to_csv_action.triggered.connect(self.save_to_csv)
        file_menu.addAction(self.save_to_csv_action)
        display_stats_action = QAction(QIcon(), 'Display statistics', self)
        display_stats_action.triggered.connect(self.display_stats)
        data_menu.addAction(display_stats_action)
        run_flowMeans_action = QAction(QIcon(), 'Run flowMeans', self)
        run_flowMeans_action.triggered.connect(self.run_flowMeans)
        data_menu.addAction(run_flowMeans_action)
        self.centroid_menu = QMenu('Centroids', self)
        self.centroid_menu.setDisabled(True)
        self.centroid_menu_action_group = QActionGroup(self.centroid_menu)
        display_centroids_action = QAction(QIcon(), 'Display centroids', self)
        display_centroids_action.setCheckable(True)
        display_centroids_action.setChecked(False)
        display_centroids_action.triggered.connect(self.plot_centroids)
        hide_centroids_action = QAction(QIcon(), 'Hide centroids', self)
        hide_centroids_action.setCheckable(True)
        hide_centroids_action.setChecked(True)
        hide_centroids_action.triggered.connect(self.hide_centroids)
        self.centroid_menu.addAction(display_centroids_action)
        self.centroid_menu.addAction(hide_centroids_action)
        self.centroid_menu_action_group.addAction(display_centroids_action)
        self.centroid_menu_action_group.addAction(hide_centroids_action)
        vis_menu.addMenu(self.centroid_menu)
        self.clust_menu = QMenu('Clusters', self)
        self.clust_menu.setDisabled(True)
        vis_menu.addMenu(self.clust_menu)
        self.clear_brush_action = QAction(QIcon(), 'Clear selection', self)
        self.clear_brush_action.setDisabled(True)
        self.clear_brush_action.triggered.connect(self.clear_brush)
        vis_menu.addAction(self.clear_brush_action)
        about_action = QAction(QIcon(), 'Show', self)
        about_action.triggered.connect(self.show_documentation)
        about_menu.addAction(about_action)

        #vis_menu.addAction(display_centroids_action)
        #vis_menu.addAction(hide_centroids_action)

        # set URL for web view to localhost for communication with flask server
        web_view.setUrl(QUrl(ROOT_URL))

    def closeEvent(self, event):
        """

        :param event:
        :return:
        """
        if self.stats_dlg is not None:
            self.stats_dlg.done(0)
        if self.doc_dlg is not None:
            self.doc_dlg.done(0)
        event.accept()

    def open_file(self, label):
        """
        Opens a .fcs file and reads the data with help of the fcsparser module
        (https://github.com/eyurtsev/fcsparser). It constructs a fcsData object holding the data read.

        In case a file has been opened already, user is asked to confirm opening the new file.

        Following the read process of the file, the user is asked to select the parameters to use for
        the analysis.
        """
        if self.pcp_displayed:
            # if a file has been opened already, check if user wants to overwrite this
            dlg = ConfirmFileOverwriteDialog()
            if not dlg.exec_():
                # if the user does NOT press ok
                return  # return immediately and do not open new file

            if self.stats_dlg is not None:
                self.stats_dlg.done(0)

        filename, _ = QFileDialog.getOpenFileName(self, 'Open file', '', 'Flow Cytometry Standard 3.0 (*.fcs)')
        # _ to catch the file type returned from getOpenFileName as separate string

        if filename:
            try:
                self.dataset = fcsData(filename)
            except:
                # catch any errors arising from fcsparser failing to open file
                title = 'Open file failed'
                message = "Your file couldn't be opened. Please note, this program currently only supports FCS 3.0. <br>" \
                          "For more details on supported files please refer to the documentation."
                dlg = ErrorDialog(title=title, message=message)
                dlg.exec_()
            self.filename = filename # if file can be opened, store filename for later use
            par = []
            for channel in self.dataset.channels['$PnN']:  # assumption about .fcs file structure
                par.append(channel)
            dlg = ChooseParametersDialog(par)
            dlg_close = dlg.exec_()
            if dlg_close == 1:
                # if user chooses values, process them by extracting checkbox states
                checkboxes = dlg.getCheckboxes()
                par_for_analysis = {}
                for checkbox in checkboxes:
                    name = checkbox.text()
                    val = checkbox.isChecked()
                    par_for_analysis[name] = val
                if self.dataset.setParametersForAnalysis(par_for_analysis): # only if selection can be written to fcsdata object
                    data_as_dict = self.dataset.returnSelectedDimensions()
                    if self.pcp_displayed:
                        # if a file has been opened already, emit a different signal to clear existing plot
                        sio.emit('overwrite plot', data={ 'data' : data_as_dict })
                        # and clear the cluster highlight menu
                        self.clust_menu.clear()
                        self.clust_menu.setDisabled(True)
                        self.centroid_menu.setDisabled(True)
                        self.clear_brush_action.setDisabled(False)
                        self.save_to_csv_action.setDisabled(False)
                    else:
                        sio.emit('initial plot', data={ 'data' : data_as_dict })
                        self.clear_brush_action.setDisabled(False)
                        self.save_to_csv_action.setDisabled(False)
                        self.pcp_displayed = True
            else:
                msg = "You didn't select any parameters for analysis. <br>" \
                      "Please open the file again select some parameters."
                title = "No parameters selected"
                dlg = ErrorDialog(title=title, message=msg)
                dlg.exec_()
        else:
            title = 'Open file failed'
            message = "Your file couldn't be opened. Please note, this program currently only supports FCS 3.0. <br>" \
                      "For more details on supported files please refer to the documentation."
            dlg = ErrorDialog(title=title, message=message)
            dlg.exec_()

    def save_to_csv(self):
        """
        Launches the dialog explaining the save file procedure. If this dialog is closed using the OK (accept) button,
        a dialog will be launched to get filepath and name to use for saving the file.
        """
        dlg = SaveFileExplDialog(self.dataset.discarded, self.dataset.discarded_count)
        if dlg.exec_():
            filename, _ = QFileDialog.getSaveFileName(self, "Save file to .csv", "", "CSV (*.csv)")
            if filename:
                try:
                    self.dataset.toCSV(filename)
                except Exception as e:
                    ttl = "Saving file failed"
                    msg = "An error occured while trying to save your file." \
                          "Please try again or contact developers if error persists."
                    dlg = ErrorDialog(title=ttl, message=msg)
                    dlg.exec_()

    def display_stats(self):
        """
        This function will launch a dialog displaying information about the data being analysed. This includes
            - general information about the data (file name, total number of events, etc)
            - details about a manual selection if applied (min, max, mean, number of events in selection)
            - details about the clustering solution if calculated (min, max, mean, standard deviation, etc of each cluster)
        """
        # if no file has been opened yet, display an error
        if self.dataset is None:
            title = "No data set"
            msg = "You have not yet opened a file."
            dlg = ErrorDialog(title=title, message=msg)
            dlg.exec_()
            return

        # otherwise, prepare arguments to be passed to DisplayStatisticsDialog
        if self.dataset.clustering is None:
            clustering = False
            clust_stats = None
        else:
            clustering = True
            clust_stats = self.dataset.getClusterStats()

        general_stats = self.dataset.getGeneralStats()
        brush_stats = self.dataset.getSelectionStats()
        selected_dim_names = self.dataset.returnSelectedDimensionsLabels()

        self.stats_dlg = StatisticsDialog(filename=self.filename,
                                          clustering=clustering,
                                          general_stats=general_stats,
                                          selected_dimensions=selected_dim_names,
                                          brush_stats=brush_stats,
                                          discarded=self.dataset.discarded_count,
                                          clust_stats=clust_stats)
        self.stats_dlg.connectUpdateStats(self)
        self.stats_dlg.show()

    def show_wait_dlg(self):
        """
        Launch the dialog that is being displayed while flowMeans runs; set it to modal to allow no other interactin
        with GUI.
        """
        self.wait_dlg = FlowMeansRunningDialog()
        self.wait_dlg.setModal(True)
        self.wait_dlg.show()

    def close_wait_dlg(self):
        """
        Close the dialog that is being displayed while flowMeans runs.
        """
        self.wait_dlg.allow_close()
        self.wait_dlg.close()

    def run_flowMeans(self):
        """
        Open the dialogs required for executing flowMeans in the appropriate order:
            - error dialog if no file has been opened
            - dialog to obtain parameters for flowMeans
            - dialog displaying parameters chosen by user and asking to confirm execution
        """
        # check if a file has been opened already
        if self.dataset is None:
            title = "Unable to execute flowMeans"
            msg = "It appears that you have not yet opened a .fcs file. <br>" \
                      "Please open a file before running the clustering algorithm."
            dlg = ErrorDialog(title=title, message=msg)
            dlg.exec_()
        else:
            # retrieve value of k from user
            get_param_dlg = GetParametersDialog()
            if get_param_dlg.exec_():
                # only continue if user exits dialogue via OK-button
                k = get_param_dlg.getK()
                discard_outliers = get_param_dlg.getDiscardFlag()
                # and run flowMeans from separate dialog that only closes once algorithm executed (i.e. method returned)
                flow_means_dlg = LaunchFlowMeansDialog(k=k, discardFlag=discard_outliers)
                if flow_means_dlg.exec_():
                    # once user confirmed to run flowMeans, dispatch worker
                    flow_means_worker = FlowMeansWorker(self.dataset, k, discard_outliers)
                    flow_means_worker.signals.started.connect(self.show_wait_dlg)
                    flow_means_worker.signals.finished.connect(self.close_wait_dlg)
                    flow_means_worker.signals.success.connect(self.process_flowMeans_result)
                    flow_means_worker.signals.error.connect(self.display_error_message)
                    if self.stats_dlg is not None:
                        self.stats_dlg.done(0) # close stats dialog while flowMeans is running
                    self.threadpool.start(flow_means_worker)

    def process_flowMeans_result(self):
        """
        Process the results obtained from a successful execution of flowMeans.
            1. re-render plot if any data points have been discarded
            2. add the highlighting layer with cluster centroids and populate the menu item for
               highlighting individual clusters
            3. enable menu options related to cluster information
            4. show dialog summarising flowMeans results
        """
        if self.dataset.discarded:
            # if data points have been discarded, plot the entire PCP again
            data_as_dict = self.dataset.returnSelectedDimensions()
            sio.emit('overwrite plot', data={ 'data' : data_as_dict })
            self.dataset.discarded = True

        # add menu options to highlight clusters
        self.highlight_clust_action_group = QActionGroup(self.clust_menu)
        self.clust_menu.clear()
        for ind in range(0, self.dataset.clust_no+1):
            if ind == 0:
                highlight_clust_action = QAction(QIcon(), "Hide cluster", self)
            else:
                highlight_clust_action = QAction(QIcon(), "Highlight cluster {}".format(ind), self)

            highlight_clust_action.setCheckable(True)
            highlight_clust_action.setChecked(False)
            highlight_clust_action.triggered.connect(lambda state, i=ind, highlight_clust_action=highlight_clust_action:
                                                     self.plot_cluster(i, highlight_clust_action))
            self.clust_menu.addAction(highlight_clust_action)
            self.highlight_clust_action_group.addAction(highlight_clust_action)

        self.plot_centroids()
        self.clust_menu.setDisabled(False)
        self.centroid_menu.setDisabled(False)

        results_dlg = ResultsDialog(clusters=self.dataset.clust_no, discarded=self.dataset.discarded,
                                    discarded_count=self.dataset.discarded_count)
        results_dlg.exec_()

    def plot_centroids(self):
        """
        Emit the socket.io signal for displaying centroids with the relevant data.
        Display error message in case of no data or clustering solution being available.
        """

        if self.dataset is not None:
            data = self.dataset.getCentroidsForPlot()
            if data: # will be False if no centroids are available
                sio.emit('show centroids', data={'data' : data})
                # as the above will clear existing highlighting layer, set option for cluster highlight
                self.highlight_clust_action_group.actions()[0].setChecked(True)
                self.centroid_menu_action_group.actions()[0].setChecked(True)
            else:
                title = "No clustering available"
                msg = "It seems that no clustering information is available. <br>" \
                      "Please run the flowMeans algorithm before displaying centroids."
                dlg = ErrorDialog(title=title, message=msg)
                dlg.exec_()
        else:
            title = "No data available"
            msg = "It seems that no data is available. <br>" \
                  "Please open a .fcs file and run the flowMeans algorithm before displaying centroids."
            dlg = ErrorDialog(title=title, message=msg)
            dlg.exec_()

    def hide_centroids(self):
        """
        Emit socket.io event for hiding highlighting layer with centroids.
        """
        sio.emit('hide centroids')

    def plot_cluster(self, i, action):
        """
        Emit socket.io event for displaying the three lines (min, max, mean) characterising a cluster.
        Requires the index of the cluster to be displayed (i).
        """
        # check state of action
        if action.isChecked():
            if i == 0:
                sio.emit('hide cluster')
            else:
                # get data for cluster visualisation
                self.centroid_menu_action_group.actions()[1].setChecked(True)
                data = self.dataset.getClusterForPlot(i-1)
                # highlight cluster
                sio.emit('show cluster', data={'data' : data})

    def apply_brush(self, data):
        """
        Process a manual selection applied trough the parallel coordinate plot. Should be called from
        the socket.io event handler for brushing. Expects the data selected as argument to pass this on
        to the fcsData instance associated with the main window.
        """
        brushed_data = pd.DataFrame(data)
        self.dataset.applySelection(brushed_data)
        if self.stats_dlg is not None:
            brush_stats = self.dataset.getSelectionStats()
            # self.stats_dlg.updateBrushTab(brush_stats=brush_stats)
            self.updateBrushStats.emit(brush_stats)

    def clear_brush(self):
        """
        Emit the socket.io event to remove all brushes.
        """
        sio.emit('clear brushes')

    def axes_reordered(self):
        """
        Ensure that all highlighting layers are hidden and appropriate menu items set to True if
        axes are reordered. Should be called from socket.io event handler for axes reorder event.
        """
        sio.emit('hide centroids')
        sio.emit('hide cluster')
        self.highlight_clust_action_group.actions()[0].setChecked(True)
        self.centroid_menu_action_group.actions()[1].setChecked(True)

    def show_documentation(self):
        """
        Show the documentation dialog and/or move it to front.
        """
        self.doc_dlg.show()
        self.doc_dlg.raise_()

    def display_error_message(self, error):
        """
        Display an error using the generic ErrorDialog. Formats information about the error.
        """
        title = str(error[0])
        msg = "An error of type <b>{}</b> occurred. <br><hr><br>" \
              "<i>Further details available are:</i> <br> {}".format(error[0], error[1][0])
        dlg = ErrorDialog(title=title, message=msg)
        dlg.exec_()


class FlowMeansWorker(QRunnable):
    """
    Thread based on QRunnable to run flowMeans. Requires FlowMeansWorkerSignals to communicate with
    remaining application.
    """
    def __init__(self, fcsdata, k, discardFlag):
        super(FlowMeansWorker, self).__init__()
        self.fcsdata = fcsdata
        self.k = k
        self.discardFlag = discardFlag
        self.signals = FlowMeansWorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            self.signals.started.emit()
            result = self.fcsdata.flowMeans(k=self.k, discard_small_clusters=self.discardFlag)
            self.signals.success.emit(result)
        except ClusterMemberError as e:
            print('Single Cluster Member Error in thread')
            self.signals.error.emit( (type(e).__name__, e.args) )
        except FailedDistanceCalculationError as e:
            print('Failed Distance Calculation Error in thread')
            self.signals.error.emit((type(e).__name__, e.args))
        except np.linalg.LinAlgError as e:
            print('LinAlgError in thread')
            self.signals.error.emit( (type(e).__name__, e.args) )
        except Exception as e:
            print('Unexpected error')
            self.signals.error.emit((type(e).__name__, e.args))
        finally:
            self.signals.finished.emit()


class FlowMeansWorkerSignals(QObject):
    """
    Signals required to interact with FlowMeansWorker (thread).
    """
    started = pyqtSignal()
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    success = pyqtSignal(bool)


class ErrorDialog(QDialog):
    """
    Generic error dialog, which will display a message passed to it.
    """
    # can't be of fixed size as length of error messages varies
    def __init__(self, title='', message='', *args, **kwargs):
        super(ErrorDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle(title)

        btn = QDialogButtonBox.Ok
        self.buttonBox = QDialogButtonBox(btn)
        self.buttonBox.accepted.connect(self.accept)

        self.label = QLabel(message)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class ConfirmFileOverwriteDialog(QDialog):
    """
    Dialog to display when user wants to open a new file if a file is currently open.
    """
    def __init__(self,  *args, **kwargs):
        super(ConfirmFileOverwriteDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("Overwrite current file")

        message = "You are trying to open a new file. <br>" \
                  "Any existing manipulations of the current <br> data set will be <b>lost</b>."
        self.label = QLabel(message)

        btn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(btn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)

        w = 300
        h = 200
        self.setFixedSize(w, h)


class ChooseParametersDialog(QDialog):
    """
    Dialog to allow user to select the dimensions extracted from FCS file that should be used
    for analysis.
    """
    def __init__(self, parameters, *args, **kwargs):
        super(ChooseParametersDialog, self).__init__(*args, **kwargs)

        self.allSelected = False

        self.setWindowTitle('Choose parameters for analysis')

        layout = QVBoxLayout()
        label = QLabel('The parameters displayed have been extracted from the .fcs file you opened. <hr>'
                       'Please choose the parameters you want to include in the analysis below. If your data contains '
                       'a time or event number attribute, you should <b>not select</b> this as it will significantly affect '
                       'the performance of the clustering algorithm provided by the software.')
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignJustify)
        layout.addWidget(label)

        self.checkboxes = []
        for parameter in parameters:
            checkbox = QCheckBox()
            checkbox.setText(parameter)
            self.checkboxes.append(checkbox)
            layout.addWidget(checkbox, alignment=Qt.AlignLeft)

        self.select_all_btn = QPushButton('Select all')
        self.select_all_btn.clicked.connect(self.selectAll)
        layout.addWidget(self.select_all_btn)

        hr = QLabel("<hr>")
        layout.addWidget(hr)

        btn = QDialogButtonBox.Ok #| QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(btn)
        self.buttonBox.accepted.connect(self.accept)
        #self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)

        self.setLayout(layout)

        # decide size based on number of parameters and set size to fixed
        h = 285 + 20 * len(parameters)
        w = 350
        self.setFixedSize(w, h)

    def selectAll(self):
        if self.allSelected:
            self.allSelected = False
            self.select_all_btn.setText('Select all')
            for checkbox in self.checkboxes:
                checkbox.setChecked(False)
        else:
            self.allSelected = True
            self.select_all_btn.setText('Deselect all')
            for checkbox in self.checkboxes:
                checkbox.setChecked(True)

    def getCheckboxes(self):
        return self.checkboxes


class GetParametersDialog(QDialog):
    """
    Dialog to display at the start of flowMeans execution. This dialog is used to obtain the parameters
    required for executing flowMeans from the user.
    """
    def __init__(self, *args, **kwargs):
        super(GetParametersDialog, self).__init__(*args, **kwargs)

        # set default value for k
        self.k_value = 20
        self.discard_flag = False
        min_k = 5
        max_k = 40

        self.setWindowTitle('Choose parameters for flowMeans')

        k_expl = "Please choose a value of <b>k</b> for the <i>flowMeans</i> algorithm. " \
               "Here, <b>k</b> is the maximum number of clusters that are expected in the data. <br>" \
               "For more details, please refer to the documentation."
        self.explanation_of_k = QLabel(k_expl)
        self.explanation_of_k.setWordWrap(True)
        self.explanation_of_k.setAlignment(Qt.AlignJustify)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_k,max_k)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setValue(self.k_value)
        self.slider.valueChanged.connect(self.setK)
        self.slider.setStyleSheet("QSlider {border: solid;}")

        self.min_label = QLabel(str(min_k))
        self.current_label = QLabel('<b>k = {}</b>'.format(str(self.k_value)))
        self.current_label.setMaximumWidth(50)
        self.max_label = QLabel(str(max_k))

        discard_lbl = "<hr>" \
                      "When you are running the <i>flowMeans</i> algorithm, it might find very small clusters " \
                      "for which the number of data points in the cluster is smaller than the number of dimensions " \
                      "you are considering. If this is the case, the algorithm will not be able to successfully " \
                      "terminate and return no solution. You can remove outliers yourself or let the algorithm do it " \
                      "based on cluster size. For more details please refer to the documentation. " \
                      "<hr> Do you want the <i>flowMeans</i> algorithm to automatically discard outliers?"
        discard_txt = "<b>Discard outliers automatically</b>"
        self.explanation_of_discard = QLabel(discard_lbl)
        self.explanation_of_discard.setWordWrap(True)
        self.explanation_of_discard.setAlignment(Qt.AlignJustify)
        self.choose_discard_layout = QHBoxLayout()
        self.discard_checkbox = QCheckBox()
        self.choose_discard_layout.addWidget(self.discard_checkbox)
        self.choose_discard_layout.addWidget(QLabel(discard_txt), 1)

        btn = QDialogButtonBox.Ok
        self.buttonBox = QDialogButtonBox(btn)
        self.buttonBox.accepted.connect(self.accept)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.explanation_of_k)
        self.slider_layout = QHBoxLayout()
        self.slider_layout.addWidget(self.min_label)
        self.slider_layout.addWidget(self.slider)
        self.slider_layout.addWidget(self.max_label)
        self.layout.addLayout(self.slider_layout)
        self.current_layout = QHBoxLayout()
        self.current_layout.addWidget(self.current_label)
        self.current_layout.setAlignment(Qt.AlignCenter)
        self.layout.addLayout(self.current_layout)
        self.layout.addWidget(self.explanation_of_discard)
        self.layout.addLayout(self.choose_discard_layout)
        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.buttonBox)
        self.btn_layout.setAlignment(Qt.AlignRight)
        self.layout.addLayout(self.btn_layout)
        self.setLayout(self.layout)

        w = 400
        h = 450
        self.setFixedSize(w, h)

    def setK(self, val):
        self.k_value = val
        self.current_label.setText('<b>k = {}</b>'.format(str(val)))

    def getK(self):
        return self.k_value

    def getDiscardFlag(self):
        return self.discard_checkbox.isChecked()


class SaveFileExplDialog(QDialog):
    """
    Dialog to display when data should be saved to CSV. Explains how save procedure operates.
    """
    def __init__(self, discarded, discarded_count, *args, **kwargs):
        super(SaveFileExplDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("Save file to .csv")

        self.layout = QVBoxLayout()

        expl = "<b>Saving your entire data set to .csv<b><br>"
        lbl = QLabel(expl)
        lbl.setAlignment(Qt.AlignHCenter)
        self.layout.addWidget(lbl)
        expl = "This will save your entire data set (all parameters, all data points) to a .csv file." \
               "Selections applied or Clustering determined cannot be saved."
        lbl = QLabel(expl)
        lbl.setAlignment(Qt.AlignJustify)
        lbl.setWordWrap(True)
        self.layout.addWidget(lbl)

        if discarded:
            expl = "<hr>" \
                   "<b>{}</b> data points have been discarded during the analysis. These will <b>not</b> " \
                   "be saved to file.".format(discarded_count)
            lbl = QLabel(expl)
            lbl.setAlignment(Qt.AlignJustify)
            lbl.setWordWrap(True)
            self.layout.addWidget(lbl)

        btn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(btn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        self.layout.addWidget(self.buttonBox)

        self.setLayout(self.layout)


class LaunchFlowMeansDialog(QDialog):
    """
    Dialog to be displayed after parameter selection for flowMeans; this dialog will show the parameters
    selected by the user to allow the user to confirm his choice.
    """
    def __init__(self, k, discardFlag, *args, **kwargs):
        super(LaunchFlowMeansDialog, self).__init__(*args, **kwargs)

        expl = "Please press <b>Run</b> to execute the clustering algorithm.<br><br>" \
               "Note that <i>this can take up to 5 minutes</i>, depending on the value of <b>k</b> you chose and " \
               "the number of events in your data. <br>" \
               "At the current development stage, it is suggested that you only use this feature for data sets " \
               "with up to 100,000 events. <br>" \
               "More information about the clustering algorithm, flowMeans, is available in the documentation. <hr>"

        explanation = QLabel(expl)
        explanation.setWordWrap(True)
        explanation.setAlignment(Qt.AlignJustify)

        k_text = "You chose <b>k={}</b>.".format(k)
        self.setWindowTitle('Run flowMeans')

        if discardFlag:
            discard_txt = "You chose to discard any outliers automatically."
        else:
            discard_txt = "You chose not to discard any outliers automatically."

        run_btn = QPushButton("Run")
        run_btn.clicked.connect(self.accept)

        self.layout = QVBoxLayout()
        self.layout.addWidget(explanation)
        self.layout.addWidget(QLabel(k_text))
        self.layout.addWidget(QLabel(discard_txt))
        self.layout.addSpacing(5)
        self.layout.addWidget(run_btn)
        self.setLayout(self.layout)

        w = 550
        h = 270
        self.setFixedSize(w, h)


class FlowMeansRunningDialog(QDialog):
    """
    Dialog to display while flowMeans is executing.
    """
    def __init__(self, *args, **kwargs): # data, k,
        super(FlowMeansRunningDialog, self).__init__(*args, **kwargs)

        msg = "<b>Please wait</b>, flowMeans is currently executing. <br>" \
              "This dialog will close automatically once the clustering has been completed. <br>" \

        self.layout = QVBoxLayout()
        self.layout.addWidget(QLabel(msg))
        self.setLayout(self.layout)

        w = 500
        h = 125
        self.setFixedSize(w, h)

        self.setWindowFlags(Qt.FramelessWindowHint)

        self._allow_close = False

    def closeEvent(self, evnt):
        if self._allow_close:
            super(FlowMeansRunningDialog, self).closeEvent(evnt)
        else:
            evnt.ignore()

    def keyPressEvent(self, evnt):
        if evnt.key() == Qt.Key_Escape:
            evnt.ignore()

    def allow_close(self):
        self._allow_close = True


class ResultsDialog(QDialog):
    """
    Dialog to display when flowMeans has successfully terminated. The dialog provides information about the
    solution found (number of clusters found, number of data points discarded if any have been discarded).
    """
    def __init__(self, clusters, discarded, discarded_count, *args, **kwargs):
        super(ResultsDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("flowMeans successfully terminated")
        if discarded:
            expl = "flowMeans successfully terminated and found <b> {} clusters</b>. <br>" \
                   "<b>{}</b> data points have been discarded during the analysis." \
                   "<br><hr>".format(clusters, discarded_count)
        else:
            expl = "flowMeans successfully terminated and found <b> {} clusters</b>. <hr>".format(clusters)
        self.layout = QVBoxLayout()
        self.layout.addWidget(QLabel(expl))
        expl = "The clusters found are represented in the parallel coordinate plot using the corresponding " \
               "centroids. Please refer to the documentation for more details."
        expl_lbl = QLabel(expl)
        expl_lbl.setWordWrap(True)
        expl_lbl.setAlignment(Qt.AlignJustify)
        self.layout.addWidget(expl_lbl)

        btn = QDialogButtonBox.Ok
        buttonBox = QDialogButtonBox(btn)
        buttonBox.accepted.connect(self.accept)
        self.layout.addWidget(buttonBox)

        self.setLayout(self.layout)
        self.setFixedSize(500, 200)


class StatisticsDialog(QDialog):
    """
    Dialog to display details about an opened data file in general, the manual selection if it has been applied
    through the visualisation using parallel coordinate plot and the clustering solution if one has been
    determined.
    """
    def __init__(self, filename, general_stats, selected_dimensions, brush_stats,
                 clustering, discarded, clust_stats, *args, **kwargs):
        super(StatisticsDialog, self).__init__(*args, **kwargs)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)
        # general window properties
        self.setWindowTitle("Details about data set")
        self.setModal(False)
        self.layout = QVBoxLayout()
        # set up tabs
        self.tabs = QTabWidget()
        self.general_tab = QWidget()
        self.brush_tab = QWidget()
        self.cluster_tab = QWidget()
        self.tabs.addTab(self.general_tab, "General")
        self.tabs.addTab(self.brush_tab, "Selected Data")
        self.tabs.addTab(self.cluster_tab, "Clustering")
        # add TabWidget to window
        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        # populate individual tabs
        # general tab --------------------------------------------------------------------------------------------------
        self.general_tab.layout = QVBoxLayout()
        filename_layout = QHBoxLayout()
        filename_lbl = QLabel(filename)
        filename_layout.addWidget(QLabel("<b>File:</b>"))
        filename_layout.addWidget(filename_lbl)
        self.general_tab.layout.addLayout(filename_layout)
        self.general_tab.layout.addWidget(QLabel("<hr>"))
        general_table = QVBoxLayout()
        for key, val in general_stats.items():
            stats_item = QHBoxLayout()
            stats_item.setAlignment(Qt.AlignTop)
            key_lbl = "<b>{}:</b>".format(key)
            val_lbl = str(val)
            stats_item.addWidget(QLabel(key_lbl))
            stats_item.addWidget(QLabel(val_lbl))
            general_table.addLayout(stats_item)
            hr = QLabel("<hr>")
            general_table.addWidget(hr)
        self.general_tab.layout.addLayout(general_table)
        self.general_tab.setLayout(self.general_tab.layout)

        # selection tab ------------------------------------------------------------------------------------------------
        self.brush_tab.layout = QVBoxLayout()
        # add the no brush message
        msg = "<b>No brush applied or brush includes all events.</b><hr>"
        self.no_brush_lbl = QLabel(msg)
        self.brush_tab.layout.addWidget(self.no_brush_lbl)
        # add the event number line
        self.event_no_key_lbl = QLabel("<b>Events:</b><hr>")
        self.event_no_key_lbl.setAlignment(Qt.AlignVCenter)
        self.event_no_lbl = QLabel("")
        self.event_no_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.total_event_no_lbl = (QLabel("(of {})".format(general_stats["Events"])))
        self.total_event_no_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        event_no_layout = QHBoxLayout()
        event_no_layout.addWidget(self.event_no_key_lbl)
        event_no_layout.addWidget(self.event_no_lbl)
        event_no_layout.addWidget(self.total_event_no_lbl)
        self.brush_tab.layout.addLayout(event_no_layout)
        # add a horizontal rule to be displayed between these and the stats of the brush
        #self.brush_tab.layout.addWidget(QLabel("<hr>"))

        # create QTableView and model
        self.brush_table = QTableView()
        self.brush_table.setAlternatingRowColors(True)
        self.brush_model = QStandardItemModel()
        self.brush_model.setHorizontalHeaderLabels(['min', 'max', 'mean'])
        self.brush_model.setVerticalHeaderLabels(selected_dimensions)
        self.brush_table.setModel(self.brush_model)
        # set table layout
        h_header = self.brush_table.horizontalHeader()
        h_header.setSectionResizeMode(QHeaderView.Fixed)
        v_header = self.brush_table.verticalHeader()
        v_header.setSectionResizeMode(QHeaderView.Fixed)
        w = self.brush_table.verticalHeader().width() + 2  # +4 seems to be needed
        for i in range(self.brush_model.columnCount()):
            w += self.brush_table.columnWidth(i)  # seems to include gridline (on my machine)
        h = self.brush_table.horizontalHeader().height() + 2
        for i in range(self.brush_model.rowCount()):
            h += self.brush_table.rowHeight(i)
        self.brush_table.setMaximumSize(QSize(w, h))
        self.brush_table.setMinimumSize(QSize(w, h))
        # add the table to the brush_tab
        self.brush_tab.layout.addWidget(self.brush_table)

        # if brush has been applied already
        self.updateBrushTab(brush_stats)

        # finally, set the layout
        self.brush_tab.setLayout(self.brush_tab.layout)

        # cluster tab --------------------------------------------------------------------------------------------------
        self.cluster_tab.layout = QVBoxLayout()
        if clustering:
            clust_no_lbl = QLabel("<b>{}</b> clusters have been found by <i>flowMeans</i>.<hr>"
                                  .format(len(clust_stats.keys())))
            self.cluster_tab.layout.addWidget(clust_no_lbl)
            if discarded > 0:
                discarded_lbl = QLabel("<b>{}</b> points have been discarded during the clustering.<hr>"
                                       .format(discarded))
                self.cluster_tab.layout.addWidget(discarded_lbl)
            # create tab for each cluster
            clust_tabs = QTabWidget()
            for k in clust_stats.keys():  # for each cluster
                # create a new tab
                tab = QWidget()
                tab.layout = QVBoxLayout()

                # display no of cluster members
                event_no_key_lbl = QLabel("<b>Events:</b>")
                event_no_key_lbl.setAlignment(Qt.AlignVCenter)
                event_no_lbl = QLabel("{}".format(clust_stats[k].pop("Events")))
                event_no_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                total_event_no_lbl = (QLabel("(of {})".format(general_stats["Events"])))
                total_event_no_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                event_no_layout = QHBoxLayout()
                event_no_layout.addWidget(event_no_key_lbl)
                event_no_layout.addWidget(event_no_lbl)
                event_no_layout.addWidget(total_event_no_lbl)
                tab.layout.addLayout(event_no_layout)

                # generate table for cluster stats
                clust_table = QTableView()
                clust_table.setAlternatingRowColors(True)
                clust_model = QStandardItemModel()
                clust_model.setHorizontalHeaderLabels(['min', 'max', 'mean', 'std'])
                clust_model.setVerticalHeaderLabels(clust_stats[k].keys())
                clust_table.setModel(clust_model)
                # set table layout
                h_header = clust_table.horizontalHeader()
                h_header.setSectionResizeMode(QHeaderView.Fixed)
                v_header = clust_table.verticalHeader()
                v_header.setSectionResizeMode(QHeaderView.Fixed)
                w = clust_table.verticalHeader().width() + 2  # +4 seems to be needed
                for i in range(clust_model.columnCount()):
                    w += clust_table.columnWidth(i)  # seems to include gridline (on my machine)
                h = clust_table.horizontalHeader().height() + 2
                for i in range(clust_model.rowCount()):
                    h += clust_table.rowHeight(i)
                clust_table.setMaximumSize(QSize(w, h))
                clust_table.setMinimumSize(QSize(w, h))
                row_count = 0  # to add new row for each dimension
                for key, val in clust_stats[k].items():
                    if key in general_stats["Channels"]:
                        dim_min = QStandardItem("{:6.2f}".format(val.min))
                        dim_max = QStandardItem("{:6.2f}".format(val.max))
                        dim_mean = QStandardItem("{:6.2f}".format(val.mean))
                        dim_std = QStandardItem("{:6.2f}".format(val.std))
                        clust_model.setItem(row_count, 0, dim_min)
                        clust_model.setItem(row_count, 1, dim_max)
                        clust_model.setItem(row_count, 2, dim_mean)
                        clust_model.setItem(row_count, 3, dim_std)
                        row_count += 1

                tab.layout.addWidget(clust_table)
                tab.setLayout(tab.layout)
                clust_tabs.addTab(tab, "Cluster {}".format(k))

            self.cluster_tab.layout.addWidget(clust_tabs)
        else:
            # if no clustering has been performed yet
            clust_lbl = QLabel("<b>No clustering has been performed yet.</b>")
            self.cluster_tab.layout.addWidget(clust_lbl)
        self.cluster_tab.setLayout(self.cluster_tab.layout)

    @pyqtSlot(dict)
    def updateBrushTab(self, brush_stats):
        if brush_stats.pop("Brushed"): # check if brush has been applied and remove entry from dictionary
            events_no = brush_stats.pop("Events")
            self.event_no_lbl.setText(str(events_no))

            self.brush_model = QStandardItemModel()
            self.brush_model.setHorizontalHeaderLabels(['min', 'max', 'mean'])
            self.brush_model.setVerticalHeaderLabels(brush_stats.keys())
            self.brush_table.setModel(self.brush_model)
            row_count = 0
            for key, val in brush_stats.items():
                dim_min = QStandardItem("{:6.2f}".format(val.min))
                dim_max = QStandardItem("{:6.2f}".format(val.max))
                dim_mean = QStandardItem("{:6.2f}".format(val.mean))
                self.brush_model.setItem(row_count, 0, dim_min)
                self.brush_model.setItem(row_count, 1, dim_max)
                self.brush_model.setItem(row_count, 2, dim_mean)
                row_count += 1

            self.no_brush_lbl.hide()
            self.event_no_key_lbl.show()
            self.event_no_lbl.show()
            self.total_event_no_lbl.show()
            self.brush_table.show()
        else:
            self.no_brush_lbl.show()
            self.event_no_key_lbl.hide()
            self.event_no_lbl.hide()
            self.total_event_no_lbl.hide()
            self.brush_table.hide()

    def connectUpdateStats(self, mw):
        mw.updateBrushStats.connect(self.updateBrushTab)


class DocumentationDialog(QDialog):
    """
    Dialog displaying multiple HTML documents with the documentation. The HTML documents to be displayed
    should be stored in the docs directory inside the application directory.

    Currently included is documentation about
       - application in general (general.html)
       - flowMeans in more detail (flowMeans.html)
       - the visualisation method Parallel Coordinate Plot (visualisation.html)
       - details about the developer (about.html)
    """
    def __init__(self, *args, **kwargs):
        super(DocumentationDialog, self).__init__(*args, **kwargs)

        self.setWindowTitle("Documentation")
        self.setModal(False)
        self.layout = QVBoxLayout()

        self.doc_tabs = QTabWidget()
        self.general_tab = QWidget()
        self.vis_tab = QWidget()
        self.flowMeans_tab = QWidget()
        self.about_tab = QWidget()

        self.doc_tabs.addTab(self.general_tab, "General")
        self.doc_tabs.addTab(self.vis_tab, "Visualisation")
        self.doc_tabs.addTab(self.flowMeans_tab, "flowMeans")
        self.doc_tabs.addTab(self.about_tab, "About")

        self.layout.addWidget(self.doc_tabs)
        self.setLayout(self.layout)

        # populate tabs
        doc_files = {"general" : None, "visualisation" : None, "flowMeans" : None, "about" : None}
        for file in doc_files.keys():
            try:
                fname = "{}/{}.html".format(resource_path("docs"), file)
                f = open(fname, 'r')
                doc_files[file] = f.read()
            except Exception as e:
                dlg = ErrorDialog(e)
                dlg.exec_()

        gen_webview = DocHtmlView()
        gen_webview.setHtml(doc_files["general"])
        self.general_tab.layout = QVBoxLayout()
        self.general_tab.layout.addWidget(gen_webview)
        self.general_tab.setLayout(self.general_tab.layout)
        vis_webview = DocHtmlView()
        vis_webview.setHtml(doc_files["visualisation"])
        self.vis_tab.layout = QVBoxLayout()
        self.vis_tab.layout.addWidget(vis_webview)
        self.vis_tab.setLayout(self.vis_tab.layout)
        flowMeans_webview = DocHtmlView()
        flowMeans_webview.setHtml(doc_files["flowMeans"])
        self.flowMeans_tab.layout = QVBoxLayout()
        self.flowMeans_tab.layout.addWidget(flowMeans_webview)
        self.flowMeans_tab.setLayout(self.flowMeans_tab.layout)
        about_webview = DocHtmlView()
        about_webview.setHtml(doc_files["about"])
        self.about_tab.layout = QVBoxLayout()
        self.about_tab.layout.addWidget(about_webview)
        self.about_tab.setLayout(self.about_tab.layout)


class WebEnginePage(QWebEnginePage):
    def acceptNavigationRequest(self, url,  _type, isMainFrame):
        if _type == QWebEnginePage.NavigationTypeLinkClicked:
            QDesktopServices.openUrl(url);
            return False
        return True


class DocHtmlView(QWebEngineView):
    def __init__(self, *args, **kwargs):
        QWebEngineView.__init__(self, *args, **kwargs)
        self.setPage(WebEnginePage(self))


class FlaskThread(QThread):
    def __init__(self, application):
        QThread.__init__(self)
        self.application = application

    def __del__(self):
        self.wait()

    def run(self):
        self.application.run(threaded=True)


if __name__ == '__main__':
    PORT = 5000
    ROOT_URL = 'http://localhost:{}'.format(PORT)

    sys.argv.append('--disable-web-security') # disable security option preventing loading/execution of local JS
    # sys.argv.append('--remote-debugging-port=5001')
    app = QApplication(sys.argv)
    window = MainWindow()

    # define socketio server and flask app
    sio = socketio.Server(async_mode='threading')
    # flask_app = Flask(__name__) # now being defined at the top
    flask_app.wsgi_app = socketio.Middleware(sio, flask_app.wsgi_app)

    @flask_app.route('/')
    def index():
        return render_template('index.html')


    @sio.on('connect')
    def connect(sid, environ):
        pass

    @sio.on('data brushed')
    def data_brushed(sid, data):
        # receive data selected by user
        window.apply_brush(data)

    @sio.on('axes reordered')
    def axes_reordered(sid):
        window.axes_reordered()

    webapp = FlaskThread(flask_app)
    webapp.start()

    app.aboutToQuit.connect(webapp.terminate)

    window.show()
    app.exec_()
