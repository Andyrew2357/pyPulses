/*
 * C++ binding for PCM1704 DCbox control. This isn't really necessary, but I
 * implemented it trying to solve an issue that wasn't there, and it happened to
 * be marginally faster, so I'm sticking with it.
 */

#include <pybind11/pybind11.h>
#include <NIDAQmx.h>

#include <string>
#include <thread>
#include <chrono>
#include <stdexcept>

namespace py = pybind11;

static void checkDAQmx(int32 error)
{
    if (error < 0)
    {
        char errBuff[2048] = {0};
        DAQmxGetExtendedErrorInfo(errBuff, sizeof(errBuff));
        throw std::runtime_error(std::string("NI-DAQmx error: ") + errBuff);
    }
}

class PCM1704Driver
{
    private:
        static constexpr uInt8 SCLK  = 0x01;
        static constexpr uInt8 SDATA = 0x02;
        static constexpr uInt8 AD2   = 0x04;
        static constexpr uInt8 AD1   = 0x08;
        static constexpr uInt8 AD0   = 0x10;
        static constexpr uInt8 WCE1  = 0x20;
        static constexpr uInt8 WCE0  = 0x40;

        TaskHandle task_out = nullptr;
        std::string dev_name;
        int change_delay;

    public:
        PCM1704Driver(const std::string& device_name, int change_delay_us)
        : dev_name (device_name)
        , change_delay (change_delay_us)
        {
            int32 error = 0;

            // Create and start the output task
            error = DAQmxCreateTask("", &task_out);
            checkDAQmx(error);

            std::string out_chan = dev_name + "/port0/line0:7";
            error = DAQmxCreateDOChan(
                task_out, out_chan.c_str(), "", 
                DAQmx_Val_ChanForAllLines
            );

            if (error < 0)
            {
                DAQmxClearTask(task_out);
                checkDAQmx(error);
            }

            error = DAQmxStartTask(task_out);
            checkDAQmx(error);
        }

        ~PCM1704Driver() 
        {
            if (task_out)
            {
                DAQmxStopTask(task_out);
                DAQmxClearTask(task_out);
            }
        }

        void set_bits(int channel, uint32_t bits)
        {
            if (channel < 0 || channel > 7)
                throw std::invalid_argument("Channel must be 0..7");

            std::vector<uInt8> wf;
            wf.reserve(79); // 79 writes

            // address bits + WCE1 high
            uInt8 state = ((channel & 0x1) ? AD0 : 0) |
                          ((channel & 0x2) ? AD1 : 0) |
                          ((channel & 0x4) ? AD2 : 0) |
                          WCE1;

            for (int i = 0; i < 24; ++i)
            {
                if (i == 1) {
                    state &= ~(WCE0 | WCE1);
                    state |= WCE0;
                }

                // SDATA
                if (i == 23) state &= ~SDATA;
                else if ((bits >> (23 - i)) & 1u) state |= SDATA;
                else state &= ~SDATA;

                // SDATA stable with SCLK low
                wf.push_back(state);

                // Pulse SCLK
                state |= SCLK; wf.push_back(state);
                state &= ~SCLK; wf.push_back(state);
            }

            // Latch
            state &= ~(WCE0 | WCE1);
            state |= WCE1;
            wf.push_back(state);

            // Two extra SCLK pulses
            state |= SCLK; wf.push_back(state);
            state &= ~SCLK; wf.push_back(state);
            state |= SCLK; wf.push_back(state);
            state &= ~SCLK; wf.push_back(state);

            // single DAQmx write
            int32 written = 0;
            int32 error = DAQmxWriteDigitalU8(
                task_out,
                wf.size(),     // number of samples
                1,             // autoStart
                10.0,          // timeout
                DAQmx_Val_GroupByChannel,
                wf.data(),
                &written,
                NULL
            );
            checkDAQmx(error);
        }
};

// Bindings
PYBIND11_MODULE(pcm1704_driver, m)
{
    m.doc() = "Minimal PCM1704 driver with persistance task management";

    py::class_<PCM1704Driver>(m, "PCM1704Driver")
        .def(py::init<const std::string&, int>(), 
            py::arg("dev_name"), 
            py::arg("change_delay"))
        .def("set_bits", &PCM1704Driver::set_bits,
            py::arg("channel"),
            py::arg("bits"),
            "Set 24-bit value on a specified channel");
}