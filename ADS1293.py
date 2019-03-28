"""Module for easier work with the TI ADS1293
   (c) stefan dot huber at stusta dot de
"""
from spidev import SpiDev
import logging
import statistics

# GPIO pin assignment on the Raspberry Pi 3
#MOSI = 19
#MISO = 21
#SCLK = 23
#GND = 25
#CE0 = 24
#CE1 = 26

F_SCLK = 20 * 1000 * 1000 # 20MHz
SAMPLE_RATE = 853

BW = 175 # bandwidth of the signal
ADC_MAX = 0xB964F0 # from the ADS1293 datasheet, p33
V_REF = 2.40 # Reference Voltage generated by the chip
V_TEST = V_REF / 12 # datasheet, p15

ADC_OUT_POS = (+(3.5 * V_TEST) / (2 * V_REF) + (1/2)) * ADC_MAX # 7846875
ADC_OUT_NEG = (-(3.5 * V_TEST) / (2 * V_REF) + (1/2)) * ADC_MAX # 4303125
ADC_OUT_ZERO = ADC_MAX/2

# The following constants are from the ADS1293 datasheet, chapter 8.6 Register Maps, pages 41ff
# Operation Mode Registers
CONFIG        = 0x00 # Main Configuration
PWR_DOWN      = 0x04 # Power-down mode
STANDBY       = 0x02 # Standby mode
START_CON     = 0x01 # Start conversion
# Input Channel Selection Registers
FLEX_CH1_CN   = 0x01 # Flex Routing Switch Control for Channel 1
TST_POS       = 0x40 # Connect Channel to positive test signal
TST_NEG       = 0x80 # Connect Channel to negative test signal
TST_ZER       = 0xC0 # Connect Channel to zero test signal
POS_EMP       = 0x00 # Positive terminal is disconnected
POS_IN1       = 0x08 # Positive terminal connected to input IN1
POS_IN2       = 0x10 # Positive terminal connected to input IN2
POS_IN3       = 0x18 # Positive terminal connected to input IN3
POS_IN4       = 0x20 # Positive terminal connected to input IN4
POS_IN5       = 0x28 # Positive terminal connected to input IN5
POS_IN6       = 0x30 # Positive terminal connected to input IN6
NEG_EMP       = 0x00 # Negative terminal is disconnected
NEG_IN1       = 0x01 # Negative terminal connected to input IN1
NEG_IN2       = 0x02 # Negative terminal connected to input IN2
NEG_IN3       = 0x03 # Negative terminal connected to input IN3
NEG_IN4       = 0x04 # Negative terminal connected to input IN4
NEG_IN5       = 0x05 # Negative terminal connected to input IN5
NEG_IN6       = 0x06 # Negative terminal connected to input IN6
FLEX_CH2_CN   = 0x02 # Flex Routing Switch Control for Channel 2
FLEX_CH3_CN   = 0x03 # Flex Routing Switch Control for Channel 3
# for automated tests
FLEX_CHx_CN = [FLEX_CH1_CN, FLEX_CH2_CN, FLEX_CH3_CN]
FLEX_PACE_CN  = 0x04 # Flex Routing Switch Control for Pace Channel
FLEX_VBAT_CN  = 0x05 # Flex Routing Switch for Battery Monitoring
VBAT_MONI_CH3 = 0x04 # Battery monitor configuration for channel 3
VBAT_MONI_CH2 = 0x02 # Battery monitor configuration for channel 2
VBAT_MONI_CH1 = 0x01 # Battery monitor configuration for channel 1
# Lead-off Detect Control Registers
LOD_CN        = 0x06 # Lead-Off Detect Control
ACAD_LOD      = 0x10 # AC analog/digital lead-off mode select (0:digital, 1:analog)
SHDN_LOD      = 0x08 # Shut down lead-off detection
SELAC_LOD     = 0x04 # Lead-off detect operation mode (0:DC, 1:AC)
ACLVL_LOD_1   = 0x00 # Programmable comparator trigger level for AC lead-off detection Level 1
ACLVL_LOD_1   = 0x01 # Programmable comparator trigger level for AC lead-off detection Level 2
ACLVL_LOD_1   = 0x02 # Programmable comparator trigger level for AC lead-off detection Level 3
ACLVL_LOD_1   = 0x03 # Programmable comparator trigger level for AC lead-off detection Level 4
LOD_EN        = 0x07 # Lead-Off Detect Enable
EN_LOD_6      = 0x20 # Enable lead-off detection for IN6
EN_LOD_5      = 0x10 # Enable lead-off detection for IN5
EN_LOD_4      = 0x08 # Enable lead-off detection for IN4
EN_LOD_3      = 0x04 # Enable lead-off detection for IN3
EN_LOD_2      = 0x02 # Enable lead-off detection for IN2
EN_LOD_1      = 0x01 # Enable lead-off detection for IN1
LOD_CURRENT   = 0x08 # Lead-Off Detect Current in steps of 8nA (0uA .. 2.040uA)
LOD_AC_CN     = 0x09 # AC Lead-Off Detect Control
ACDIV_FACTOR  = 0x80 # AC lead off test frequency divisor factor (0: K=1, 1: K=16)
# Common-Mode Detection and Right-Leg Drive Feedback Control Registers
CMDET_EN      = 0x0A # Common-Mode Detect Enable
CMDET_CN      = 0x0B # Common-Mode Detect Control
RLD_CN        = 0x0C # Right-Leg Drive Control
# Wilson Control Registers
WILSON_EN1    = 0x0D # Wilson Reference Input one Selection
WILSON_EN2    = 0x0E # Wilson Reference Input two Selection
WILSON_EN3    = 0x0F # Wilson Reference Input three Selection
WILSON_CN     = 0x10 # Wilson Reference Control
# Referenve Registers
# OSC Control Registers
OSC_CN        = 0x12 # Clock Source and Output Clock Control
# AFE Control Registers
AFE_RES       = 0x13 # Analog Front End Frequency and Resolution
AFE_SHDN_CN   = 0x14 # Analog Front End Shutdown Control
AFE_SHDN_INA_CH1 = 0x01
AFE_SHDN_INA_CH2 = 0x02
AFE_SHDN_INA_CH3 = 0x04
AFE_SHDN_SDM_CH1 = 0x08
AFE_SHDN_SDM_CH2 = 0x10
AFE_SHDN_SDM_CH3 = 0x20
AFE_SHDN_INA_CHx = [AFE_SHDN_INA_CH1, AFE_SHDN_INA_CH2, AFE_SHDN_INA_CH3]
AFE_SHDN_SDM_CHx = [AFE_SHDN_SDM_CH1, AFE_SHDN_SDM_CH2, AFE_SHDN_SDM_CH3]
AFE_FAULT_CN  = 0x15 # Analog Front End Fault DEtection Control
#RESERVED     = 0x16 # -
AFE_PACE_CN   = 0x17 # Analog Pace Channel Output Routing Control
# Error Status Registers
ERROR_LOD     = 0x18 # Lead-Off Detect Error Status
ERROR_STATUS  = 0x19 # Other Error Status
ERROR_RANGE_1 = 0x1A # Channel 1 AFE Out-of-Range Status
ERROR_RANGE_2 = 0x1B # Channel 2 AFE Out-of-Range Status
ERROR_RANGE_3 = 0x1C # Channel 3 AFE Out-of-Range Status
ERROR_SYNC    = 0x1D # Synchronization Error
ERROR_MISC    = 0x1E # Miscellaneous Errors
# Digital Registers
DIGO_STRENGTH = 0x1F # Digital Output Strength
R2_RATE       = 0x21 # R2 Decimation Rate
R3_RATE_CH1   = 0x22 # R3 Decimation Rate for Channel 1
R3_RATE_CH2   = 0x23 # R3 Decimation Rate for Channel 2
R3_RATE_CH3   = 0x24 # R3 Decimation Rate for Channel 3
R3_RATE_CHx   = [R3_RATE_CH1, R3_RATE_CH2, R3_RATE_CH3]
R1_RATE       = 0x25 # R1 Decimation Rate
DIS_EFILTER   = 0x26 # ECG Filter Disable
DRDYB_SRC     = 0x27 # Data Ready Pin Source
DRDYB_SRC_P1  = 0x01 # DRDYB driven by CH1 pace
DRDYB_SRC_P2  = 0x02 # DRDYB driven by CH1 pace
DRDYB_SRC_P3  = 0x04 # DRDYB driven by CH1 pace
DRDYB_SRC_E1  = 0x08 # DRDYB driven by CH1 pace
DRDYB_SRC_E2  = 0x10 # DRDYB driven by CH1 pace
DRDYB_SRC_E3  = 0x20 # DRDYB driven by CH1 pace
SYNCB_CN      = 0x28 # SYNCB In/Out Pin Control
MASK_DRDYB    = 0x29 # Optional Mask Control for DRDYB Output
MASK_ERROR    = 0x2A # Mask Error on ALARMB Pin
#RESERVED     = 0x2B # -
#RESERVED     = 0x2C # -
#RESERVED     = 0x2D # -
ALARM_FILTER  = 0x2E # Digital Filter for Analog Alarm Signals
CH_CNFG       = 0x2F # Configure Channel for Loop Read Back Mode
STS_EN        = 0x01 # Enable DATA_STATUS read back
P1_EN         = 0x02 # Enable DATA_CH1_PACE read back
P2_EN         = 0x04 # Enable DATA_CH2_PACE read back
P3_EN         = 0x08 # Enable DATA_CH3_PACE read back
E1_EN         = 0x10 # Enable DATA_CH1_ECG read back
E2_EN         = 0x20 # Enable DATA_CH2_ECG read back
E3_EN         = 0x40 # Enable DATA_CH3_ECG read back
Px_EN         = [P1_EN, P2_EN, P3_EN]
Ex_EN         = [E1_EN, E2_EN, E3_EN]
# Pace and ECG Data Read Back Registers
DATA_STATUS   = 0x30 # ECG and Pace Data Ready Status
E3_DRDY       = 0x80 # Channel 3 ECG data ready
E2_DRDY       = 0x40 # Channel 2 ECG data ready
E1_DRDY       = 0x20 # Channel 1 ECG data ready
Ex_DRDY       = [E1_DRDY, E2_DRDY, E3_DRDY]
P3_DRDY       = 0x10 # Channel 3 pace data ready
P2_DRDY       = 0x08 # Channel 2 pace data ready
P1_DRDY       = 0x04 # Channel 1 pace data ready
Px_DRDY       = [P1_DRDY, P2_DRDY, P3_DRDY]
ALARMB        = 0x02 # ALARMB status
#RESERVED     = 0x01
DATA_CH1_PACE = 0x31 # Channel 1 Pace Data, 1st register
#               0x32 # Channel 1 Pace Data, 2nd register
DATA_CH2_PACE = 0x33 # Channel 2 Pace Data, 1st register
#               0x34 # Channel 2 Pace Data, 2nd register
DATA_CH2_PACE = 0x35 # Channel 3 Pace Data, 1st register
#               0x36 # Channel 3 Pace Data, 2nd register
DATA_CH1_ECG  = 0x37 # Channel 1 ECG Data, 1st register
#             = 0x38 # Channel 1 ECG Data, 2nd register
#             = 0x39 # Channel 1 ECG Data, 3rd register
DATA_CH2_ECG  = 0x3A # Channel 2 ECG Data, 1st register
#             = 0x3B # Channel 2 ECG Data, 2nd register
#             = 0x3C # Channel 2 ECG Data, 3rd register
DATA_CH3_ECG  = 0x3D # Channel 3 ECG Data, 1st register
#             = 0x3E # Channel 3 ECG Data, 2nd register
#             = 0x3F # Channel 3 ECG Data, 3rd register
REVID         = 0x40 # Revision ID
DATA_LOOP     = 0x50 # Loop Ready-Back Address

NUM_PACE_DATA_REGISTERS = 2
NUM_ECG_DATA_REGISTERS = 3

READ_BIT  = 0x80 # OR with the register bit
WRITE_BIT = 0x7F # AND with the register bit

class ADS1293:
    """SpiDev Wrapper for TI ADS1293
    """

    def __init__(self, bus=0, device=0):
        """initializes the SPI bus for the ADS1293
        """
        self.bus, self.device = bus, device
        self.spi = SpiDev()
        self.open()
        self.inax_max = [0, 0, 0]
        self.inax_min = [0, 0, 0]
        self.inax_zero = [0, 0, 0]

    def open(self):
        """opens the SPI bus for the ADS1293
        """
        self.spi.open(self.bus, self.device)
        self.spi.max_speed_hz = F_SCLK

    def close(self):
        self.spi.close()

    def read(self, register, num_bytes=1):
        """reads a number of bytes from the ADS1293
        """
        data = [0] * (num_bytes + 1)
        data[0] = READ_BIT | register
        self.spi.xfer2(data)
        return data[1:]

    def write(self, register, value):
        """writes the value value to register register
        """
        data = [WRITE_BIT & register, value]
        self.spi.xfer2(data)

    def go_to_standby(self):
        """sends the ADS1293 to standby
        """
        self.write(CONFIG, STANDBY)

    def test(self):
        # initial setup
        count = 100
        self.write(OSC_CN, 0x04) # set external oscillator
        self.write(R2_RATE, 0x02) # set R2 decimation rate as 5
        for i in range(3): # for all three Channels
            # shut down unused circuitry
            shdn = 0
            for j in range(3):
                if i != j:
                    shdn = shdn | AFE_SHDN_INA_CHx[j] | AFE_SHDN_SDM_CHx[j]
            self.write(AFE_SHDN_CN, shdn)

            # initialize channel
            self.write(R3_RATE_CHx[i], 0x02) # set R3 decimation rate as 6
            self.write(CH_CNFG, Ex_EN[i]) # enable loopback reading

            # positive test voltage
            self.write(FLEX_CHx_CN[i], TST_POS) # connect channel to pos test signal
            self.write(CONFIG, START_CON) # start data conversion
            results = self.stat_values(count, Ex_DRDY[i])
            self.write(CONFIG, STANDBY) # send chip to standby
            self.inax_max[i] = results[0]
            logging.info(
                "Channel %d positive test:\n\t"
                "Expected: %.1f\n\t"
                "Received: %.1f avg [%.1f]\n\t\t"
                "  (%.1f min, %.1f max, %.1f stddev)" %
                (i+1, ADC_OUT_POS, results[0], results[0] - ADC_OUT_POS,
                 results[1], results[2], results[3])
            )

            # negative test voltage
            self.write(FLEX_CHx_CN[i], TST_NEG) # connect channel to neg test signal
            self.write(CONFIG, START_CON) # start data conversion
            results = self.stat_values(count, Ex_DRDY[i])
            self.write(CONFIG, STANDBY) # send chip to standby
            self.inax_min[i] = results[0]
            logging.info(
                "Channel %d negative test:\n\t"
                "Expected: %.1f\n\t"
                "Received: %.1f avg [%.1f]\n\t\t"
                "  (%.1f min, %.1f max, %.1f stddev)" %
                (i+1, ADC_OUT_NEG, results[0], results[0] - ADC_OUT_NEG,
                 results[1], results[2], results[3])
            )

            # zero test voltage
            self.write(FLEX_CHx_CN[i], TST_ZER)
            self.write(CONFIG, START_CON) # start data conversion
            results = self.stat_values(count, Ex_DRDY[i])
            self.write(CONFIG, STANDBY) # send chip to standby
            self.inax_zero[i] = results[0]
            logging.info(
                "Channel %d zero test:\n\t"
                "Expected: %.1f\n\t"
                "Received: %.1f avg [%.1f]\n\t\t"
                "  (%.1f min, %.1f max, %.1f stddev)" %
                (i+1, ADC_OUT_ZERO, results[0], results[0] - ADC_OUT_ZERO,
                 results[1], results[2], results[3])
            )

            # deinitialize channel
            self.write(FLEX_CHx_CN[i], 0x00) # disconnect the channel
            self.write(R3_RATE_CHx[i], 0x00) # reset R3 decimation rate

        # reset the self
        self.reset()

    def reset(self):
        self.deinit_three_lead_ecg()
        self.deinit_five_lead_ecg()

    def check_ready(self, datareadybit=E1_DRDY):
        result = self.read(DATA_STATUS)
        return result[0] & datareadybit

    def stat_values(self, count=100, channel=E1_DRDY):
        """returns a list of avg, min, max, stdev
        """
        results = []
        i = 0
        while i < count:
            if self.check_ready(channel):
                data = self.read(DATA_LOOP, NUM_ECG_DATA_REGISTERS * 1)
                results.append(data[0] << 16 | data[1] << 8 | data[2])
                i += 1
        return [
            sum(results)/len(results),
            min(results),
            max(results),
            statistics.stdev(results)]

    def init_three_lead_ecg(self):
        """Initializes the ADS1293 to the 3-Lead ECG application from the
        datasheet, chapter 9.2.1
        """
        # 1. Set address 0x01 = 0x11: Connect channel 1’s INP to IN2 and INN to IN1.
        self.write(FLEX_CH1_CN, POS_IN2 | NEG_IN1)
        # 2. Set address 0x02 = 0x19: Connect channel 2’s INP to IN3 and INN to IN1.
        self.write(FLEX_CH2_CN, POS_IN3 | NEG_IN1)
        # 3. Set address 0x0A = 0x07: Enable the common-mode detector on input pins
        # IN1, IN2 and IN3.
        self.write(CMDET_EN, 0x07)
        # 4. Set address 0x0C = 0x04: Connect the output of the RLD amplifier
        # internally to pin IN4.
        self.write(RLD_CN, 0x04)
        # 5. Set address 0x12 = 0x04: Use external crystal and feed the internal
        # oscillator's output to the digital.
        self.write(OSC_CN, 0x04)
        # 6. Set address 0x14 = 0x24: Shuts down unused channel 3’s signal path.
        self.write(AFE_SHDN_CN, 0x24)
        # 7. Set address 0x21 = 0x02: Configures the R2 decimation rate as 5 for all
        # channels.
        self.write(R2_RATE, 0x02)
        # 8. Set address 0x22 = 0x02: Configures the R3 decimation rate as 6 for
        # channel 1.
        self.write(R3_RATE_CH1, 0x02)
        # 9. Set address 0x23 = 0x02: Configures the R3 decimation rate as 6 for
        # channel 2.
        self.write(R3_RATE_CH2, 0x02)
        # 10. Set address 0x27 = 0x08: Configures the DRDYB source to channel 1 ECG
        # (or fastest channel).
        self.write(DRDYB_SRC, 0x08)
        # 11. Set address 0x2F = 0x30: Enables channel 1 ECG and channel 2 ECG for
        # loop read-back mode.
        self.write(CH_CNFG, 0x30)
        # 12. Set address 0x00 = 0x01: Starts data conversion.
        self.write(CONFIG, START_CON)

    def deinit_three_lead_ecg(self):
        """Deinitializes the ADS1293.
        Effectively this sends the ADS1293 to standby and sets the initialized
        registers to their default values.
        """
        # send ADS1293 to standby mode to save energy
        self.write(CONFIG, STANDBY)
        # reset registers to their default values
        self.write(CH_CNFG, 0x00)
        self.write(DRDYB_SRC, 0x00)
        self.write(R3_RATE_CH2, 0x00)
        self.write(R3_RATE_CH1, 0x00)
        self.write(R2_RATE, 0x00)
        self.write(AFE_SHDN_CN, 0x00)
        self.write(OSC_CN, 0x00)
        self.write(RLD_CN, 0x00)
        self.write(CMDET_EN, 0x00)
        self.write(FLEX_CH2_CN, 0x00)
        self.write(FLEX_CH1_CN, 0x00)

    def init_five_lead_ecg(self):
        """Initializes the ADS1293 to the 5-Lead ECG application from the
        datasheet, chapter 9.2.2
        """
        # 1. Set address 0x01 = 0x11: Connect channel 1’s INP to IN2 and INN to IN1.
        self.write(FLEX_CH1_CN, POS_IN2 | NEG_IN1)
        # 2. Set address 0x02 = 0x19: Connect channel 2’s INP to IN3 and INN to IN1.
        self.write(FLEX_CH2_CN, POS_IN3 | NEG_IN1)
        # 3. Set address 0x03 = 0x2E: Connect channel 3’s INP to IN5 and INN to IN6.
        self.write(FLEX_CH3_CN, POS_IN5 | NEG_IN6)
        # 4. Set address 0x0A = 0x07: Enable the common-mode detector on input pins IN1, IN2 and IN3.
        self.write(CMDET_EN, 0x07)
        # 5. Set address 0x0C = 0x04: Connect the output of the RLD amplifier internally to pin IN4.
        self.write(RLD_CN, 0x04)
        # 6. Set addresses 0x0D = 0x01, 0x0E = 0x02, 0x0F = 0x03: COnnects the first buffer of the Wilson reference to the IN1 pin, the second buffer to the IN2 pin, and the third buffer to the IN3 pin.
        self.write(WILSON_EN1, 0x01)
        self.write(WILSON_EN2, 0x02)
        self.write(WILSON_EN3, 0x03)
        # 7. Set address 0x10 = 0x01: Connects the output of the Wilson reference internally to IN6
        self.write(WILSON_CN, 0x01)
        # 8. Set address 0x12 = 0x04: Uses external crystal and feeds the output of the internal oscillator module to the digital.
        self.write(OSC_CN, 0x04)
        # 9. Set address 0x21 = 0x02: Configures the R2 decimation rate as 5 for all channels.
        self.write(R2_RATE, 0x02)
        # 10. Set address 0x22 = 0x02: Configures the R3 decimation rate as 6 for channel 1.
        self.write(R3_RATE_CH1, 0x02)
        # 11. Set address 0x23 = 0x02: Configures the R3 decimation rate as 6 for channel 2.
        self.write(R3_RATE_CH2, 0x02)
        # 12. Set address 0x24 = 0x02: Configures the R3 decimation rate as 6 for channel 3.
        self.write(R3_RATE_CH3, 0x02)
        # 12. Set address 0x27 = 0x08: Configures the DRDYB source to channel 1 ECG (or fastest channel).
        self.write(DRDYB_SRC, 0x08)
        # 11. Set address 0x2F = 0x70: Enables ECG channel 1, ECG channel 2 and ECG channel 3 for loop read-back mode.
        self.write(CH_CNFG, 0x70)
        # 12. Set address 0x00 = 0x01: Starts data conversion.
        self.write(CONFIG, START_CON)

    def deinit_five_lead_ecg(self):
        """Deinitializes the ADS1293.
        Effectively this sends the ADS1293 to standby and sets the initialized
        registers to their default values.
        """
        # send ADS1293 to standby mode to save energy
        self.write(CONFIG, STANDBY)
        # reset registers to their default values
        self.write(CH_CNFG, 0x00)
        self.write(DRDYB_SRC, 0x00)
        self.write(R3_RATE_CH3, 0x00)
        self.write(R3_RATE_CH2, 0x00)
        self.write(R3_RATE_CH1, 0x00)
        self.write(R2_RATE, 0x00)
        self.write(OSC_CN, 0x00)
        self.write(WILSON_CN, 0x00)
        self.write(WILSON_EN3, 0x00)
        self.write(WILSON_EN2, 0x00)
        self.write(WILSON_EN1, 0x00)
        self.write(RLD_CN, 0x00)
        self.write(CMDET_EN, 0x00)
        self.write(FLEX_CH3_CN, 0x00)
        self.write(FLEX_CH2_CN, 0x00)
        self.write(FLEX_CH1_CN, 0x00)

if __name__ == "__main__":
    print("This module is not thought for standalone usage, it only defines the"
          "ADS1293 object class.")
