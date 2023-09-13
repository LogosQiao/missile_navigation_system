import sys
import os
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from UI.gui import Ui_Form


class TheMainWindow(QMainWindow, Ui_Form):
    def __init__(self, parent=None):
        super(TheMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.butt_input.clicked.connect(self.choose_input_file)
        self.butt_output.clicked.connect(self.choose_out_file)
        self.para_set.clicked.connect(self.setParameter)
        self.calculate.clicked.connect(self.getCalulation)
        self.plot.clicked.connect(self.plotting)
        self.loss_analysis.clicked.connect(self.analysis_loss)

    def choose_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            with open("../config/configure.json", 'r') as r:
                cfg = json.load(r)
            r.close()
            self.text_input.setText(os.path.basename(file_path))
            cfg["input_path"] = file_path
            with open("../config/configure.json", 'w') as w:
                json.dump(cfg,w)
            w.close()


    def choose_out_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", "Text Files (*.txt);;All Files (*)")
        if file_path:
            with open("../config/configure.json", 'r') as r:
                cfg = json.load(r)
            r.close()
            self.text_output.setText(os.path.basename(file_path))
            cfg["output_path"] = file_path
            with open("../config/configure.json", 'w') as w:
                json.dump(cfg, w)
            w.close()

    def setParameter(self):
        with open("../config/configure.json", 'r') as r:
            cfg = json.load(r)
        r.close()

        # Initial state error settings
        cfg['err_l'] = float(self.err_l.text()) if self.err_l.text() != "" else 0
        cfg['err_lambda'] = float(self.err_lambda.text()) if self.err_lambda.text() != "" else 0
        cfg['err_h'] = float(self.err_h.text()) if self.err_h.text() != "" else 0
        cfg['err_vn'] = float(self.err_vn.text()) if self.err_vn.text() != "" else 0
        cfg['err_vu'] = float(self.err_vu.text()) if self.err_vu.text() != "" else 0
        cfg['err_ve'] = float(self.err_ve.text()) if self.err_ve.text() != "" else 0
        cfg['err_phi'] = float(self.err_phi.text())  if self.err_phi.text() != "" else 0
        cfg['err_sigma'] = float(self.err_sigma.text()) if self.err_sigma.text() != "" else 0
        cfg['err_gamma'] = float(self.err_gamma.text()) if self.err_gamma.text() != "" else 0

        # Loading error settings
        cfg['ax_bd'] = float(self.ax_bd.text()) if self.ax_bd.text() != "" else 0
        cfg['ay_bd'] = float(self.ay_bd.text()) if self.ay_bd.text() != "" else 0
        cfg['az_bd'] = float(self.az_bd.text()) if self.az_bd.text() != "" else 0
        cfg['wx_bd'] = float(self.wx_bd.text()) if self.wx_bd.text() != "" else 0
        cfg['wy_bd'] = float(self.wy_bd.text()) if self.wy_bd.text() != "" else 0
        cfg['wz_bd'] = float(self.wz_bd.text()) if self.wz_bd.text() != "" else 0

        cfg['ax_lp'] = float(self.ax_lp.text()) if self.ax_lp.text() != "" else 0
        cfg['ay_lp'] = float(self.ay_lp.text()) if self.ay_lp.text() != "" else 0
        cfg['az_lp'] = float(self.az_lp.text()) if self.az_lp.text() != "" else 0
        cfg['wx_lp'] = float(self.wx_lp.text()) if self.wx_lp.text() != "" else 0
        cfg['wy_lp'] = float(self.wy_lp.text()) if self.wy_lp.text() != "" else 0
        cfg['wz_lp'] = float(self.wz_lp.text()) if self.wz_lp.text() != "" else 0

        cfg['ax_wdx'] = float(self.ax_wdx.text()) if self.ax_wdx.text() != "" else 0
        cfg['ay_wdx'] = float(self.ay_wdx.text()) if self.ay_wdx.text() != "" else 0
        cfg['az_wdx'] = float(self.az_wdx.text()) if self.az_wdx.text() != "" else 0
        cfg['wx_wdx'] = float(self.wx_wdx.text()) if self.wx_wdx.text() != "" else 0
        cfg['wy_wdx'] = float(self.wy_wdx.text()) if self.wy_wdx.text() != "" else 0
        cfg['wz_wdx'] = float(self.wz_wdx.text()) if self.wz_wdx.text() != "" else 0

        # Simulation program parameter settings
        cfg['start_point'] = int(self.start_point.text()) if self.start_point.text() != "" else 0
        cfg['duration'] = int(self.duration.text()) if self.duration.text() != "" else 0
        cfg['count'] = int(self.count.text()) if self.count.text() != "" else 0

        with open("../config/configure.json", 'w') as w:
            json.dump(cfg, w)
            w.close()

    def getCalulation(self):
        pass

    def plotting(self):
        pass

    def analysis_loss(self):
        pass


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TheMainWindow()
    win.show()
    sys.exit(app.exec_())