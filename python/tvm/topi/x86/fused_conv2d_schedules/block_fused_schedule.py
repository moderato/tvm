import tvm

def schedule_block_fused_nhwc(outs, stages, params, bn_relu1=None, bn_relu2=None):
    outs = [outs] if isinstance(outs, tvm.tensor.Tensor) else outs
    s = tvm.create_schedule([x.op for x in outs])

    ######## Get stages
    Input = stages[0][0]
    PaddedInput = stages[1][0]
    if bn_relu1 is not None:
        Inter, InterScaleShift, InterReLU = stages[2]
        IntermediateStage = InterReLU
        F_1, Scale_1, Shift_1 = params[1]
    else:
        Inter = stages[2][0]
        IntermediateStage = Inter
        F_1 = params[1][0]

    hasPaddedInter = False
    if bn_relu2 is not None:
        if 'Padded' in stages[3][0].op.name:
            hasPaddedInter = True
            PaddedInter = stages[3][0]
            Out, OutScaleShift, OutAdd, OutReLU = stages[4]
        else:
            Out, OutScaleShift, OutAdd, OutReLU = stages[3]
        OutputStage = OutReLU
        F_2, Scale_2, Shift_2 = params[2]
    else:
        if 'Padded' in stages[3][0].op.name:
            hasPaddedInter = True
            PaddedInter = stages[3][0]
            Out, OutAdd = stages[4]
        else:
            Out, OutAdd = stages[3]
        OutputStage = OutAdd
        F_2 = params[2][0]
 
    ######## Searchable parameters
    # --------------------
    output_step_tile_size_h = 2
    output_step_tile_size_w = 2
    step_num_h = 2
    step_num_w = 2
    reduce_split1 = 4
    reduce_split2 = 4
    input_reuse = 2
    intermediate_reuse = 4
    intermediate_block_split = 2
    output_block_split = 2
    num_thread_x = 32
    # --------------------
    output_tile_size_h = output_step_tile_size_h * step_num_h
    output_tile_size_w = output_step_tile_size_w * step_num_w
    num_thread_y = output_step_tile_size_w
    num_thread_z = output_step_tile_size_h
    # num_vthread_z = step_num_h * step_num_w
    num_vthread_z = step_num_h
    num_vthread_y = step_num_w
    num_vthread_x = 32
    num_vthread_w = 32
    num_vthread_u = 32
    # --------------------

    ######## Input data, weights, BN, etc
    # ---
    s[PaddedInput].compute_inline()
    # ---
    # IL = s.cache_read(Input, "local", [PaddedInput])
    # ---
    PaddedSharedInput = s.cache_read(PaddedInput, "shared", [Inter])
    FS_1 = s.cache_read(F_1, "shared", [Inter])
    FS_2 = s.cache_read(F_2, "shared", [Out])
    s[Inter].set_scope("local")
    ConvLocalAccumulator = Inter

    # Put the input of the second stage into the shared memory
    if hasPaddedInter:
        s[PaddedInter].set_scope("shared")
    else:
        s[IntermediateStage].set_scope("shared")

    if bn_relu1 is not None:
        s[InterScaleShift].compute_inline()
        ScaleL_1 = s.cache_read(Scale_1, "local", [InterScaleShift])
        ShiftL_1 = s.cache_read(Shift_1, "local", [InterScaleShift])
        if hasPaddedInter:
            s[IntermediateStage].compute_inline()
    IntermediateStage = PaddedInter

    if bn_relu2 is not None:
        s[OutScaleShift].compute_inline()
        s[Out].set_scope("local")
        ScaleL_2 = s.cache_read(Scale_2, "local", [OutScaleShift])
        ShiftL_2 = s.cache_read(Shift_2, "local", [OutScaleShift])
        # s[OutAdd].set_scope("local")
        # s[OutAdd].compute_inline()
        OL = OutAdd
    else:
        OL = s.cache_write(OutputStage, "local")

    ######## Blocks and threads
    block_x = tvm.thread_axis("blockIdx.x")
    thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
    thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
    thread_z = tvm.thread_axis((0, num_thread_z), "threadIdx.z")

    ######## Vthreads
    vthread_x = tvm.thread_axis((0, num_vthread_x), "vthread", name="vthread_x")
    vthread_y = tvm.thread_axis((0, num_vthread_y), "vthread", name="vthread_y")
    vthread_z = tvm.thread_axis((0, num_vthread_z), "vthread", name="vthread_z")

    return s
