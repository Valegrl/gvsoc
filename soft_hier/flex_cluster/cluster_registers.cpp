/*
 * Copyright (C) 2024 ETH Zurich and University of Bologna
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Authors: Germain Haugou, ETH Zurich (germain.haugou@iis.ee.ethz.ch)
            Yichao  Zhang , ETH Zurich (yiczhang@iis.ee.ethz.ch)
            Chi     Zhang , ETH Zurich (chizhang@iis.ee.ethz.ch)
 */

#include <vp/vp.hpp>
#include <vp/itf/io.hpp>
#include <vp/itf/wire.hpp>
#include <cpu/iss/include/offload.hpp>


#define MAX_CLUSTERS 16

class ClusterRegisters : public vp::Component
{

public:
    ClusterRegisters(vp::ComponentConf &config);

private:
    static vp::IoReqStatus req(vp::Block *__this, vp::IoReq *req);
    static vp::IoReqStatus barrier_reg_req(vp::Block *__this, vp::IoReq *req);
    static vp::IoReqStatus global_barrier_slave_req(vp::Block *__this, vp::IoReq *req);
    static void global_barrier_master_resp(vp::Block *__this, vp::IoReq *req);
    static void wakeup_event_handler(vp::Block *__this, vp::ClockEvent *event);
    static void hbm_preload_done_handler(vp::Block *__this, bool value);
    static void inst_preheat_done_handler(vp::Block *__this, bool value);
    void fetch_start_check();
    void send_wakeup_to_clusters(uint32_t value);

    vp::Trace trace;

    // Existing ports (0x40000000 periph)
    vp::IoSlave input_itf;
    vp::WireMaster<bool> barrier_ack_itf;
    vp::WireMaster<IssOffloadInsn<uint32_t>*> rocache_cfg_itf;
    vp::WireSlave<bool> hbm_preload_done_itf;
    vp::WireSlave<bool> inst_preheat_done_itf;
    vp::WireMaster<bool> fetch_start_itf;

    // Inter-cluster barrier ports (0x20000000)
    vp::IoSlave barrier_reg_input_itf;
    vp::IoMaster global_barrier_master_itf;
    vp::IoSlave global_barrier_slave_itf;

    vp::ClockEvent * wakeup_event;
    int wakeup_latency;
    bool eoc_reached;
    uint32_t hbm_preload_done;
    uint32_t inst_preheat_done;
    uint32_t fetch_started;

    // Inter-cluster barrier config
    int cluster_id;
    int num_cluster_x;
    int num_cluster_y;
    uint32_t sync_base;
    uint32_t sync_interleave;
    uint32_t sync_special_mem;

    // Pre-allocated IoReqs for wakeup messages via sync bus
    vp::IoReq wakeup_reqs[MAX_CLUSTERS];
    uint32_t wakeup_data;
};



ClusterRegisters::ClusterRegisters(vp::ComponentConf &config)
    : vp::Component(config)
{
    this->traces.new_trace("trace", &this->trace, vp::DEBUG);

    // Existing ports (periph at 0x40000000)
    this->input_itf.set_req_meth(&ClusterRegisters::req);
    this->new_slave_port("input", &this->input_itf);
    this->new_master_port("barrier_ack", &this->barrier_ack_itf);
    this->new_master_port("rocache_cfg", &this->rocache_cfg_itf);

    this->hbm_preload_done_itf.set_sync_meth(&ClusterRegisters::hbm_preload_done_handler);
    this->inst_preheat_done_itf.set_sync_meth(&ClusterRegisters::inst_preheat_done_handler);
    this->new_slave_port("hbm_preload_done", &this->hbm_preload_done_itf);
    this->new_slave_port("inst_preheat_done", &this->inst_preheat_done_itf);
    this->new_master_port("fetch_start", &this->fetch_start_itf);
    this->hbm_preload_done = 0;
    this->inst_preheat_done = 0;
    this->fetch_started = 0;

    // Inter-cluster barrier ports (0x20000000)
    this->barrier_reg_input_itf.set_req_meth(&ClusterRegisters::barrier_reg_req);
    this->new_slave_port("barrier_reg_input", &this->barrier_reg_input_itf);

    this->global_barrier_master_itf.set_resp_meth(&ClusterRegisters::global_barrier_master_resp);
    this->new_master_port("global_barrier_master", &this->global_barrier_master_itf);

    this->global_barrier_slave_itf.set_req_meth(&ClusterRegisters::global_barrier_slave_req);
    this->new_slave_port("global_barrier_slave", &this->global_barrier_slave_itf);

    this->wakeup_event = this->event_new(&ClusterRegisters::wakeup_event_handler);
    wakeup_latency = get_js_config()->get_child_int("wakeup_latency");
    eoc_reached = false;

    // Inter-cluster config
    this->cluster_id = get_js_config()->get_child_int("cluster_id");
    this->num_cluster_x = get_js_config()->get_child_int("num_cluster_x");
    this->num_cluster_y = get_js_config()->get_child_int("num_cluster_y");
    this->sync_base = get_js_config()->get_child_int("sync_base");
    this->sync_interleave = get_js_config()->get_child_int("sync_interleave");
    this->sync_special_mem = get_js_config()->get_child_int("sync_special_mem");
    this->wakeup_data = 1;
}

void ClusterRegisters::wakeup_event_handler(vp::Block *__this, vp::ClockEvent *event) {
    ClusterRegisters *_this = (ClusterRegisters *)__this;
    _this->barrier_ack_itf.sync(1);
    _this->trace.msg("Control registers wake up signal work and write %d to barrier ack output\n", 1);
}

void ClusterRegisters::hbm_preload_done_handler(vp::Block *__this, bool value)
{
    ClusterRegisters *_this = (ClusterRegisters *)__this;
    _this->hbm_preload_done = 1;
    _this->trace.msg(vp::Trace::LEVEL_DEBUG, "HBM Preloading Done\n");
    _this->fetch_start_check();
}

void ClusterRegisters::inst_preheat_done_handler(vp::Block *__this, bool value)
{
    ClusterRegisters *_this = (ClusterRegisters *)__this;
    _this->inst_preheat_done = 1;
    _this->trace.msg(vp::Trace::LEVEL_DEBUG, "Instruction Preheating Done\n");
    _this->fetch_start_check();
}

void ClusterRegisters::fetch_start_check()
{
    if (this->hbm_preload_done && this->inst_preheat_done)
    {
        if (this->fetch_started == 0)
        {
            this->fetch_started = 1;
            this->fetch_start_itf.sync(1);
        }
    }
}


vp::IoReqStatus ClusterRegisters::req(vp::Block *__this, vp::IoReq *req)
{
    ClusterRegisters *_this = (ClusterRegisters *)__this;

    uint64_t offset = req->get_addr();
    uint8_t *data = req->get_data();
    uint64_t size = req->get_size();
    bool is_write = req->get_is_write();
    int initiator = req->get_initiator();

    _this->trace.msg("Control registers access (offset: 0x%x, size: 0x%x, is_write: %d, data:%x, initiator:%d)\n", offset, size, is_write, *(uint32_t *)data, initiator);

    if (is_write && size == 4)
    {
        uint32_t value = *(uint32_t *)data;
        if (offset == 0 && (value & 1) && !_this->eoc_reached)
        {
            _this->eoc_reached = true;
            std::cout << "EOC register return value: 0x" << std::hex << ((value - 1) >> 1) << std::endl;
            _this->time.get_engine()->quit(value >> 1);
        }
        if (offset == 4 && value == 0xFFFFFFFF)
        {
            _this->event_enqueue(_this->wakeup_event, _this->wakeup_latency);
        }
        if (offset == 0x48 || offset == 0x4C || offset == 0x50 || offset == 0x54)
        {
            IssOffloadInsn<uint32_t> insn;
            insn.arg_a = (offset - 0x48) >> 2;
            insn.arg_b = 0;
            insn.arg_c = value;
            _this->rocache_cfg_itf.sync(&insn);
        }
        if (offset == 0x58 || offset == 0x5C || offset == 0x60 || offset == 0x64)
        {
            IssOffloadInsn<uint32_t> insn;
            insn.arg_a = (offset - 0x58) >> 2;
            insn.arg_b = 1;
            insn.arg_c = value;
            _this->rocache_cfg_itf.sync(&insn);
        }
    }

    return vp::IO_REQ_OK;
}


/*
 * Barrier config registers at ARCH_CLUSTER_REG_BASE (0x20000000).
 * Offsets:
 *   0:  R → cluster_id;  W → enqueue barrier_ack (wakeup local cores)
 *   4:  R → 1 (enable_value)
 *   8:  R → num_clusters (num_cluster_x * num_cluster_y)
 *  12:  R → num_cluster_x
 *  16:  R → num_cluster_y
 *  20:  W → debug barrier annotation (no-op)
 *  24:  R → 0 (disable_value)
 *  28:  W → send wakeup to clusters via sync bus
 */
vp::IoReqStatus ClusterRegisters::barrier_reg_req(vp::Block *__this, vp::IoReq *req)
{
    ClusterRegisters *_this = (ClusterRegisters *)__this;

    uint64_t offset = req->get_addr();
    uint8_t *data = req->get_data();
    uint64_t size = req->get_size();
    bool is_write = req->get_is_write();

    _this->trace.msg("Barrier config register access (offset: 0x%x, size: 0x%x, is_write: %d)\n",
                     offset, size, is_write);

    if (size == 4)
    {
        if (is_write)
        {
            uint32_t value = *(uint32_t *)data;
            switch (offset)
            {
                case 0:
                    // Wakeup local cores (enqueue barrier_ack)
                    _this->trace.msg("Barrier reg: local wakeup (cluster %d)\n", _this->cluster_id);
                    if (!_this->wakeup_event->is_enqueued())
                        _this->event_enqueue(_this->wakeup_event, _this->wakeup_latency);
                    break;
                case 20:
                    // Debug annotation (no-op)
                    _this->trace.msg("Barrier annotation: %d\n", value);
                    break;
                case 28:
                    // Send wakeup to clusters via sync bus
                    _this->trace.msg("Barrier reg: sending wakeup to clusters (value: 0x%x)\n", value);
                    _this->send_wakeup_to_clusters(value);
                    break;
                default:
                    break;
            }
        }
        else
        {
            // Read
            uint32_t value = 0;
            switch (offset)
            {
                case 0:
                    value = _this->cluster_id;
                    break;
                case 4:
                    value = 1; // enable_value
                    break;
                case 8:
                    value = _this->num_cluster_x * _this->num_cluster_y;
                    break;
                case 12:
                    value = _this->num_cluster_x;
                    break;
                case 16:
                    value = _this->num_cluster_y;
                    break;
                case 24:
                    value = 0; // disable_value
                    break;
                default:
                    break;
            }
            *(uint32_t *)data = value;
        }
    }

    return vp::IO_REQ_OK;
}


vp::IoReqStatus ClusterRegisters::global_barrier_slave_req(vp::Block *__this, vp::IoReq *req)
{
    ClusterRegisters *_this = (ClusterRegisters *)__this;

    _this->trace.msg("Global barrier slave: received wakeup from sync bus (cluster %d)\n",
                     _this->cluster_id);

    // Trigger local barrier_ack to wake all cores
    if (!_this->wakeup_event->is_enqueued())
        _this->event_enqueue(_this->wakeup_event, _this->wakeup_latency);

    return vp::IO_REQ_OK;
}


void ClusterRegisters::global_barrier_master_resp(vp::Block *__this, vp::IoReq *req)
{
    // Fire-and-forget: nothing to do on response
}


void ClusterRegisters::send_wakeup_to_clusters(uint32_t value)
{
    int num_clusters = this->num_cluster_x * this->num_cluster_y;
    uint32_t sync_size = this->sync_interleave + this->sync_special_mem;

    for (int i = 0; i < num_clusters; i++)
    {
        // Target cluster i's special_mem region via sync bus
        uint32_t target_addr = this->sync_base + i * sync_size + this->sync_interleave;

        this->wakeup_reqs[i].init();
        this->wakeup_reqs[i].set_addr(target_addr);
        this->wakeup_reqs[i].set_size(4);
        this->wakeup_reqs[i].set_is_write(true);
        this->wakeup_reqs[i].set_data((uint8_t *)&this->wakeup_data);

        this->trace.msg("Sending wakeup to cluster %d at addr 0x%x\n", i, target_addr);
        this->global_barrier_master_itf.req(&this->wakeup_reqs[i]);
    }
}


extern "C" vp::Component *gv_new(vp::ComponentConf &config)
{
    return new ClusterRegisters(config);
}