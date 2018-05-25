namespace CNTK.FSharp

module Sequential =

    open System.Collections.Generic
    open CNTK

    type Computation = DeviceDescriptor -> Variable -> Function

    type Specification = {
        Features: Variable
        Labels: Variable
        Model: Computation
        Loss: Loss
        Eval: Loss
        }

    [<RequireQualifiedAccess>]
    module Layer = 

        /// Combine 2 Computation Layers into 1
        let add (next:Computation) (curr:Computation) : Computation =
            fun device ->
                fun variable ->
                    let intermediate = curr device variable |> var
                    next device intermediate

        /// Combine a sequence of Computation Layers into 1
        let sequence (computations: Computation seq) =
            computations
            |> Seq.reduce (fun acc c -> add c acc)
        
        let scale<'T> (scalar:'T) : Computation = 
            fun device ->
                fun input ->
                    CNTKLib.ElementTimes(Constant.Scalar<'T>(scalar, device), input)

        let dense (outputDim:int) : Computation =
            fun device ->
                fun input ->
               
                    let input : Variable =
                        if (input.Shape.Rank <> 1)
                        then
                            let newDim = input.Shape.Dimensions |> Seq.reduce (*)
                            CNTKLib.Reshape(input, shape [ newDim ]) |> var
                        else input

                    let inputDim = input.Shape.[0]
                    let dataType = input.DataType

                    let weights = 
                        new Parameter(
                            shape [outputDim; inputDim], 
                            dataType,
                            CNTKLib.GlorotUniformInitializer(
                                float CNTKLib.DefaultParamInitScale,
                                CNTKLib.SentinelValueForInferParamInitRank,
                                CNTKLib.SentinelValueForInferParamInitRank, 
                                uint32 1),
                            device, 
                            "weights")

                    let product = 
                        CNTKLib.Times(weights, input, "product") 
                        |> var

                    let bias = new Parameter(shape [ outputDim ], 0.0f, device, "bias")
                    CNTKLib.Plus(bias, product)

        let dropout (proba:float) : Computation = 
            fun device ->
                fun input ->
                    CNTKLib.Dropout(input,proba)

    [<RequireQualifiedAccess>]
    module Activation = 

        let ReLU : Computation = 
            fun device ->
                fun input ->
                    CNTKLib.ReLU(input)

        let sigmoid : Computation = 
            fun device ->
                fun input ->
                    CNTKLib.Sigmoid(input)

        let tanh : Computation = 
            fun device ->
                fun input ->
                    CNTKLib.Tanh(input)

    [<RequireQualifiedAccess>]
    module Conv2D = 
    
        type Kernel = {
            Width: int
            Height: int
            }

        type Strides = {
            Horizontal: int
            Vertical: int
            }

        type Conv2D = {
            Kernel: Kernel 
            Filters: int
            Initializer: Initializer
            Strides: Strides
            }

        let conv2D = {
            Kernel = { Width = 1; Height = 1 } 
            Filters = 1
            Initializer = GlorotUniform
            Strides = { Horizontal = 1; Vertical = 1 }
            }

        let convolution (args:Conv2D) : Computation = 
            fun device ->
                fun input ->
                    let kernel = args.Kernel
                    let inputChannels = input.Shape.Dimensions.[2]
                    let convParams = 
                        device
                        |> Param.init (
                            [ kernel.Width; kernel.Height; inputChannels; args.Filters ], 
                            DataType.Float,
                            args.Initializer)

                    CNTKLib.Convolution(
                        convParams, 
                        input, 
                        shape [args.Strides.Horizontal; args.Strides.Vertical ; inputChannels]
                        )

        type Window = {
            Width: int
            Height: int          
            }

        type Pool2D = {
            Window: Window
            Strides: Strides 
            PoolingType: PoolingType
            Padding: bool
            }
                         
        let pooling (args:Pool2D) : Computation = 
            fun device ->
                fun input ->

                    let window = args.Window
                    let strides = args.Strides

                    CNTKLib.Pooling(
                        input, 
                        args.PoolingType,
                        shape [ window.Width; window.Height ], 
                        shape [ strides.Horizontal; strides.Vertical ], 
                        [| args.Padding |]
                        )

    [<RequireQualifiedAccess>]
    module Recurrent =
        let internal embedding embeddingDim : Computation =
            fun device ->
                fun input ->
                    let inputDim = input.Shape.[0]
                    let embeddingParameters = 
                        device
                        |> Param.init ([ embeddingDim; inputDim ],  DataType.Float, GlorotUniform)            
                    embeddingParameters * input

        let internal stabilize<'ElementType> : Computation =
            fun device ->
                fun input ->
                    let isFloatType = (typeof<'ElementType> = typeof<System.Single>)
    
                    let f, fInv =
                        if (isFloatType)
                        then
                            Constant.Scalar(4.0f, device),
                            Constant.Scalar(DataType.Float,  1.0 / 4.0) 
                        else
                            Constant.Scalar(4.0, device),
                            Constant.Scalar(DataType.Double, 1.0 / 4.0)
        
                    let beta = 
                        fInv.Tensor .*
                        Tensor.log(
                            Constant.Scalar(f.DataType, 1.0).Tensor +  
                            Tensor.exp(
                                f.Tensor .* 
                                (new Parameter(new NDShape(), f.DataType, 0.99537863, device) |> Tensor.from)
                                )
                            )
                            
                    beta .* input.Tensor |> Tensor.toFunction

        let internal LSTMPCellWithSelfStabilization<'ElementType>( 
            input:Variable, 
            prevOutput:Variable, 
            prevCellState:Variable, 
            device:DeviceDescriptor) : (Function * Function) =

                let outputDim = prevOutput.Shape.[0]
                let cellDim = prevCellState.Shape.[0]
        
                let isFloatType = (typeof<'ElementType> = typeof<System.Single>)

                let dataType : DataType = 
                    if isFloatType 
                    then DataType.Float 
                    else DataType.Double

                let createBiasParam : int -> Tensor =
                    fun dim ->
                        match dataType with
                        | DataType.Float -> new Parameter(shape [ dim ], 0.01f, device, "")
                        | DataType.Double -> new Parameter(shape [ dim ], 0.01, device, "")
                        |> Tensor.from
            
                // TODO: replace by a function...
                let seeder =
                    let mutable s = uint32 1
                    fun () ->
                        s <- s + uint32 1
                        s

                let createProjectionParam : int -> Tensor = 
                    fun oDim -> 
                        new Parameter(
                            shape [ oDim; NDShape.InferredDimension ],
                            dataType, 
                            CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seeder ()), 
                            device
                            )
                        |> Tensor.from
        
                let createDiagWeightParam : int -> Tensor = 
                    fun dim ->
                        new Parameter(
                            shape [ dim ], 
                            dataType, 
                            CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seeder ()), 
                            device
                            )
                        |> Tensor.from

                let stabilizedPrevOutput : Tensor = 
                    stabilize<'ElementType> device prevOutput |> Fun
                let stabilizedPrevCellState : Tensor = 
                    stabilize<'ElementType> device prevCellState |> Fun
        
                let projectInput : unit -> Tensor = 
                    fun () -> 
                        createBiasParam cellDim
                        + (createProjectionParam cellDim * input.Tensor)

                // Input gate
                let it : Tensor =
                    projectInput () 
                    + (createProjectionParam cellDim * stabilizedPrevOutput) 
                    + (createDiagWeightParam cellDim .* stabilizedPrevCellState)
                    |> Tensor.sigmoid
                               
                let bit : Tensor = 
                    it .* (projectInput () + (createProjectionParam cellDim *  stabilizedPrevOutput) |> Tensor.tanh)
                                  
                // Forget-me-not gate
                let ft : Tensor = 
                    projectInput () 
                    + (createProjectionParam cellDim * stabilizedPrevOutput)
                    + (createDiagWeightParam cellDim .* stabilizedPrevCellState)
                    |> Tensor.sigmoid
                        
                let bft : Tensor = ft .* prevCellState.Tensor

                let ct : Tensor = bft + bit

                // Output gate
                let ot : Tensor = 
                    projectInput () 
                    + (createProjectionParam cellDim * stabilizedPrevOutput) 
                    + (createDiagWeightParam cellDim .* Fun (stabilize<'ElementType> device ct.toVar))
                    |> Tensor.sigmoid
                        
                let ht : Tensor = ot .* Tensor.tanh(ct)

                let c : Tensor = ct
                let h : Tensor = 
                    if (outputDim <> cellDim) 
                    then (createProjectionParam outputDim * Fun(stabilize<'ElementType> device ht.toVar))
                    else ht

                (h.toFun, c.toFun)

        let internal LSTMPComponentWithSelfStabilization<'ElementType>(
            input:Variable,
            outputShape:NDShape, 
            cellShape:NDShape,
            recurrenceHookH:Variable -> Function,
            recurrenceHookC:Variable -> Function,
            device:DeviceDescriptor) : (Function * Function) =

                let dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes)
                let dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes)

                let LSTMCell = LSTMPCellWithSelfStabilization<'ElementType>(input, dh, dc, device)
                let actualDh = recurrenceHookH (var (fst LSTMCell))
                let actualDc = recurrenceHookC (var (snd LSTMCell))

                let replacement : IDictionary<Variable, Variable> =
                    [
                        dh, var (actualDh)
                        dc, var (actualDc)
                    ]
                    |> dict

                (fst LSTMCell).ReplacePlaceholders(replacement) |> ignore
        
                LSTMCell

        /// <summary>
        /// Build a one direction recurrent neural network (RNN) with long-short-term-memory (LSTM) cells.
        /// http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        /// </summary>
        /// <param name="input">the input variable</param>
        /// <param name="numOutputClasses">number of output classes</param>
        /// <param name="embeddingDim">dimension of the embedding layer</param>
        /// <param name="LSTMDim">LSTM output dimension</param>
        /// <param name="cellDim">cell dimension</param>
        /// <param name="device">CPU or GPU device to run the model</param>
        /// <param name="outputName">name of the model output</param>
        /// <returns>the RNN model</returns>
        let LSTMSequenceClassifierNet numOutputClasses embeddingDim LSTMDim cellDim : Computation =
            fun device ->
                fun input ->
                    let embeddingFunction : Function = embedding embeddingDim device input
                    let pastValueRecurrenceHook : (Variable -> Function) = fun x -> CNTKLib.PastValue(x)
                    let LSTMFunction, _ = 
                        LSTMPComponentWithSelfStabilization<single>(
                            var embeddingFunction,
                            shape [ LSTMDim ],
                            shape [ cellDim ],
                            pastValueRecurrenceHook,
                            pastValueRecurrenceHook,
                            device)

                    let thoughtVectorFunction : Function = CNTKLib.SequenceLast(var LSTMFunction)

                    Layer.dense numOutputClasses device (var thoughtVectorFunction)
       
    type Schedule = {
        Rate: float
        MinibatchSize: int
        }

    type Config = {
        MinibatchSize: int
        Epochs: int
        Device: DeviceDescriptor
        Schedule: Schedule
        Optimizer: Optimizer
        } 

    let learning (predictor:Function) (config:Config) = 
        let schedule = config.Schedule 
        let learningRatePerSample = 
            new TrainingParameterScheduleDouble(schedule.Rate, uint32 schedule.MinibatchSize)        
        let parameterLearners = 
            predictor.Parameters ()
            |> learnWith (config.Optimizer,learningRatePerSample)
            |> Seq.singleton
            |> ResizeArray
        parameterLearners

    type TextFormatSource = {
        FilePath:string
        Features:string
        Labels:string
        } with
        member this.Mappings (spec:Specification) : (string * TextFormat.InputMappings) =
            this.FilePath,
            {
                Features = [
                    { Variable = spec.Features; SourceName = this.Features }
                ]
                Labels = { Variable = spec.Labels; SourceName = this.Labels }
            }

    type Learner () =

        let progress = new Event<Minibatch.TrainingSummary> ()
        member this.MinibatchProgress = progress.Publish

        member this.learn 
            (source:MinibatchSource) 
            (featureStreamName:string, labelsStreamName:string) 
            (config:Config) 
            (spec:Specification) =

            let device = config.Device

            let predictor = spec.Model device spec.Features
            let loss = evaluation spec.Loss (predictor,spec.Labels)
            let eval = evaluation spec.Eval (predictor,spec.Labels)   

            let parameterLearners = learning predictor config     
            let trainer = Trainer.CreateTrainer(predictor, loss, eval, parameterLearners)
        
            let input = spec.Features
            let labels = spec.Labels
            let featureStreamInfo = source.StreamInfo(featureStreamName)
            let labelStreamInfo = source.StreamInfo(labelsStreamName)
            let minibatchSize = uint32 (config.MinibatchSize)

            let rec learnEpoch (step,epoch) = 

                Minibatch.summary trainer
                |> progress.Trigger

                if epoch <= 0
                // we are done : return function
                then predictor
                else
                    let step = step + 1
                    let minibatchData = source.GetNextMinibatch(minibatchSize, device)

                    let arguments : IDictionary<Variable, MinibatchData> =
                        [
                            input, minibatchData.[featureStreamInfo]
                            labels, minibatchData.[labelStreamInfo]
                        ]
                        |> dict

                    trainer.TrainMinibatch(arguments, device) |> ignore
                
                    // MinibatchSource is created with MinibatchSource.InfinitelyRepeat.
                    // Batching will not end. Each time minibatchSource completes an sweep (epoch),
                    // the last minibatch data will be marked as end of a sweep. We use this flag
                    // to count number of epochs.
                    let epoch = 
                        if Minibatch.isSweepEnd minibatchData
                        then epoch - 1
                        else epoch

                    learnEpoch (step, epoch)

            learnEpoch (0, config.Epochs)
