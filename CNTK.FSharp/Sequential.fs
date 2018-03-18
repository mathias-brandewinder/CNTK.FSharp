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
        let stack (next:Computation) (curr:Computation) : Computation =
            fun device ->
                fun variable ->
                    let intermediate = new Variable(curr device variable)
                    next device intermediate

        /// Combine a sequence of Computation Layers into 1
        let sequence (computations: Computation seq) =
            computations
            |> Seq.reduce (fun acc c -> stack c acc)
        
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
                            new Variable(CNTKLib.Reshape(input, shape [ newDim ]))
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
                        new Variable(CNTKLib.Times(weights, input, "product"))

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

        type Stride = {
            Horizontal: int
            Vertical: int
            }

        type Conv2D = {
            Kernel: Kernel 
            OutputFeatures: int
            Initializer: Initializer
            Strides: Stride
            }

        let conv2D = {
            Kernel = { Width = 1; Height = 1 } 
            OutputFeatures = 1
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
                            [ kernel.Width; kernel.Height; inputChannels; args.OutputFeatures ], 
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
            Stride : Stride 
            PoolingType : PoolingType
            }
                         
        let pooling (args:Pool2D) : Computation = 
            fun device ->
                fun input ->

                    let window = args.Window
                    let stride = args.Stride

                    CNTKLib.Pooling(
                        input, 
                        args.PoolingType,
                        shape [ window.Width; window.Height ], 
                        shape [ stride.Horizontal; stride.Vertical ], 
                        [| true |]
                        )
            
    type Schedule = {
        Rate: float
        MinibatchSize: int
        }

    type Config = {
        MinibatchSize: int
        Epochs: int
        Device: DeviceDescriptor
        Schedule: Schedule
        } 

    let learning (predictor:Function) (schedule:Schedule) =   
        let learningRatePerSample = 
            new TrainingParameterScheduleDouble(schedule.Rate, uint32 schedule.MinibatchSize)
        let parameterLearners =
            ResizeArray<Learner>(
                [ 
                    Learner.SGDLearner(predictor.Parameters(), learningRatePerSample) 
                ])
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

            let parameterLearners = learning predictor config.Schedule     
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
