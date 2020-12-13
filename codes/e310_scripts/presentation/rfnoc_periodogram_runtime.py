#!/usr/bin/env python2
# -*- coding: utf-8 -*-
##################################################
# GNU Radio Python Flow Graph
# Title: Rfnoc Periodogram Runtime
# Generated: Wed Jun 12 01:02:41 2019
##################################################

from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import uhd
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from optparse import OptionParser
import ettus


class rfnoc_fft_vectoriir(gr.top_block):

    def __init__(self):
        gr.top_block.__init__(self, "Rfnoc Fft Vectoriir")

        ##################################################
        # Variables
        ##################################################
        self.alpha = alpha = 0.986
        self.seconds = seconds = 5
        self.samp_rate = samp_rate = 40e6
        self.rf_gain = rf_gain = 0
        self.num_points = num_points = 512
        self.hard_coded_sample_count = hard_coded_sample_count = 2883584
        self.freq = freq = 2440e6
        self.device3 = device3 = ettus.device3(uhd.device_addr_t( ",".join(('type=e3x0', "fpga=/home/root/usrp_e310_fpga_RFNOC_sg3.bit")) ))
        self.decim_rate = decim_rate = 18
        self.beta = beta = 1-alpha

        ##################################################
        # Blocks
        ##################################################
        self.uhd_rfnoc_streamer_vector_iir_0 = ettus.rfnoc_generic(
              self.device3,
              uhd.stream_args( # TX Stream Args
                  cpu_format="fc32",
                  otw_format="sc16",
                  args="spp={},alpha={},beta={}".format(num_points, alpha, beta),
              ),
              uhd.stream_args( # TX Stream Args
                  cpu_format="fc32",
                  otw_format="sc16",
                  args="spp={},alpha={},beta={}".format(num_points, alpha, beta),
              ),
              "VectorIIR", -1, -1,
        )
        self.uhd_rfnoc_streamer_vector_iir_0.set_arg("alpha", alpha)
        self.uhd_rfnoc_streamer_vector_iir_0.set_arg("beta",  beta)

        self.uhd_rfnoc_streamer_radio_0 = ettus.rfnoc_radio(
            self.device3,
            uhd.stream_args( # Tx Stream Args
                cpu_format="fc32",
                otw_format="sc16",
                args="", # empty
            ),
            uhd.stream_args( # Rx Stream Args
                cpu_format="fc32",
                otw_format="sc16",
          args='spp=512',
            ),
            0, -1
        )
        self.uhd_rfnoc_streamer_radio_0.set_rate(samp_rate)

        self.uhd_rfnoc_streamer_radio_0.set_rx_freq(freq, 0)
        self.uhd_rfnoc_streamer_radio_0.set_rx_gain(rf_gain, 0)
        self.uhd_rfnoc_streamer_radio_0.set_rx_dc_offset(True, 0)


        self.uhd_rfnoc_streamer_radio_0.set_rx_bandwidth(40e6, 0)

        if "RX2":
            self.uhd_rfnoc_streamer_radio_0.set_rx_antenna("RX2", 0)


        self.uhd_rfnoc_streamer_radio_0.set_clock_source("internal")
        self.uhd_rfnoc_streamer_keep_one_in_n_0 = ettus.rfnoc_generic(
            self.device3,
            uhd.stream_args( # TX Stream Args
                cpu_format="fc32", # TODO: This must be made an option
                otw_format="sc16",
                args="n="+str(decim_rate),
            ),
            uhd.stream_args( # RX Stream Args
                cpu_format="fc32", # TODO: This must be made an option
                otw_format="sc16",
                args="",
            ),
            "KeepOneInN", -1, -1,
        )
        self.uhd_rfnoc_streamer_fft_0 = ettus.rfnoc_generic(
            self.device3,
            uhd.stream_args( # TX Stream Args
                cpu_format="fc32", # TODO: This must be made an option
                otw_format="sc16",
                args="spp={}".format(num_points), # Need to set the FFT size here, or it won't go into the GR IO signature
            ),
            uhd.stream_args( # RX Stream Args
                cpu_format="fc32", # TODO: This must be made an option
                otw_format="sc16",
                args="",
            ),
            "FFT", -1, -1,
        )
        self.uhd_rfnoc_streamer_fft_0.set_arg("direction", "forward")
        self.uhd_rfnoc_streamer_fft_0.set_arg("scaling", 1000)
        self.uhd_rfnoc_streamer_fft_0.set_arg("shift", "normal")
        self.uhd_rfnoc_streamer_fft_0.set_arg("magnitude_out", "COMPLEX")

        self.blocks_head_0 = blocks.head(gr.sizeof_gr_complex*1, hard_coded_sample_count)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_gr_complex*1, '/home/root/test_data', False)
        self.blocks_file_sink_0.set_unbuffered(True)



        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_head_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.uhd_rfnoc_streamer_keep_one_in_n_0, 0), (self.blocks_head_0, 0))
        self.device3.connect(self.uhd_rfnoc_streamer_fft_0.get_block_id(), 0, self.uhd_rfnoc_streamer_vector_iir_0.get_block_id(), 0)
        self.device3.connect(self.uhd_rfnoc_streamer_radio_0.get_block_id(), 0, self.uhd_rfnoc_streamer_fft_0.get_block_id(), 0)
        self.device3.connect(self.uhd_rfnoc_streamer_vector_iir_0.get_block_id(), 0, self.uhd_rfnoc_streamer_keep_one_in_n_0.get_block_id(), 0)

    def get_alpha(self):
        return self.alpha

    def set_alpha(self, alpha):
        self.alpha = alpha
        self.set_beta(1-self.alpha)
        self.uhd_rfnoc_streamer_vector_iir_0.set_arg("alpha", self.alpha)

    def get_seconds(self):
        return self.seconds

    def set_seconds(self, seconds):
        self.seconds = seconds

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_rfnoc_streamer_radio_0.set_rate(self.samp_rate)

    def get_rf_gain(self):
        return self.rf_gain

    def set_rf_gain(self, rf_gain):
        self.rf_gain = rf_gain

        self.uhd_rfnoc_streamer_radio_0.set_rx_gain(self.rf_gain, 0)

    def get_num_points(self):
        return self.num_points

    def set_num_points(self, num_points):
        self.num_points = num_points
        self.uhd_rfnoc_streamer_fft_0.set_arg("spp", self.num_points)

    def get_hard_coded_sample_count(self):
        return self.hard_coded_sample_count

    def set_hard_coded_sample_count(self, hard_coded_sample_count):
        self.hard_coded_sample_count = hard_coded_sample_count
        self.blocks_head_0.set_length(self.hard_coded_sample_count)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq

        self.uhd_rfnoc_streamer_radio_0.set_rx_freq(self.freq, 0)

    def get_device3(self):
        return self.device3

    def set_device3(self, device3):
        self.device3 = device3

    def get_decim_rate(self):
        return self.decim_rate

    def set_decim_rate(self, decim_rate):
        self.decim_rate = decim_rate

    def get_beta(self):
        return self.beta

    def set_beta(self, beta):
        self.beta = beta
        self.uhd_rfnoc_streamer_vector_iir_0.set_arg("beta", self.beta)


def main(top_block_cls=rfnoc_fft_vectoriir, options=None):

    tb = top_block_cls()
    tb.start()
    tb.wait()


if __name__ == '__main__':
    main()
