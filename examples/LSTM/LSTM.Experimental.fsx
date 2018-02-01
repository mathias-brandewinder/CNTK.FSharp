(*
https://github.com/Microsoft/CNTK/master/Examples/TrainingCSharp/Common/LSTMSequenceClassifier.cs
*)

open System
open System.IO
open System.Collections.Generic

#load "../../CNTK.fsx"
open CNTK

let MiniBatchDataIsSweepEnd(minibatchValues:seq<MinibatchData>) =
    minibatchValues 
    |> Seq.exists(fun a -> a.sweepEnd)

let FullyConnectedLinearLayer(
    input:Variable, 
    outputDim:int, 
    device:DeviceDescriptor,
    outputName:string) : Function =

    let inputDim = input.Shape.[0]

    let timesParam = 
        new Parameter(
            shape [outputDim; inputDim], 
            DataType.Float,
            CNTKLib.GlorotUniformInitializer(
                float CNTKLib.DefaultParamInitScale,
                CNTKLib.SentinelValueForInferParamInitRank,
                CNTKLib.SentinelValueForInferParamInitRank, 
                uint32 1),
            device, 
            "timesParam")

    let timesFunction = 
        Tensor.times(timesParam.Tensor, input.Tensor, "times")

    let plusParam = new Parameter(shape [ outputDim ], 0.0f, device, "plusParam")
    
    Tensor.plus(plusParam.Tensor, timesFunction, outputName) 
    |> Tensor.toFunction

let Embedding(input:Variable, embeddingDim:int, device:DeviceDescriptor) : Function =

    let inputDim = input.Shape.[0]
    let embeddingParameters = 
        new Parameter(
            shape [ embeddingDim; inputDim ], 
            DataType.Float, 
            CNTKLib.GlorotUniformInitializer(), 
            device
            )
    
    embeddingParameters.Tensor * input.Tensor 
    |> Tensor.toFunction

let Stabilize<'ElementType>(x:Variable, device:DeviceDescriptor) : Function =

    let isFloatType = (typeof<'ElementType> = typeof<System.Single>)
    
    let f, fInv =
        if (isFloatType)
        then
            Constant.Scalar(4.0f, device),
            Constant.Scalar(DataType.Float,  1.0 / 4.0) 
        else
            Constant.Scalar(4.0, device),
            Constant.Scalar(DataType.Double, 1.0 / 4.0)

    let one = Constant.Scalar(f.DataType, 1.0).Tensor

    let beta = 
        fInv.Tensor .*
        Tensor.log(
            one +  
            Tensor.exp
                (f.Tensor .* (new Parameter(new NDShape(), f.DataType, 0.99537863, device)).Tensor)                                    
            )
                
    beta .* x.Tensor |> Tensor.toFunction   

let LSTMPCellWithSelfStabilization<'ElementType>( 
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
                | DataType.Float -> new Parameter(shape [ dim ], 0.01f, device, "") |> Tensor.from
                | DataType.Double -> new Parameter(shape [ dim ], 0.01, device, "") |> Tensor.from
            
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

        let stabilizedPrevOutput : Tensor = Stabilize<'ElementType>(prevOutput, device) |> Fun
        let stabilizedPrevCellState : Tensor = Stabilize<'ElementType>(prevCellState, device) |> Fun

        let projectInput : unit -> Tensor = 
            fun () -> createBiasParam (cellDim) + (createProjectionParam(cellDim) * input.Tensor)

        // Input gate
        let it : Tensor =
            (projectInput () + (createProjectionParam (cellDim)) *  stabilizedPrevOutput)
            + (createDiagWeightParam (cellDim) .* stabilizedPrevCellState)
            |> Tensor.sigmoid
             
        let bit : Tensor = 
            it .* Tensor.tanh (projectInput () + (createProjectionParam(cellDim) * stabilizedPrevOutput))
                                         
        // Forget-me-not gate
        let ft : Tensor =            
            projectInput () + 
            (createProjectionParam(cellDim) *  stabilizedPrevOutput) +
            (createDiagWeightParam(cellDim) .* stabilizedPrevCellState)
            |> Tensor.sigmoid
                        
        let bft : Tensor = ft .* Var prevCellState 

        let ct : Tensor = bft + bit

        // Output gate
        let ot : Tensor = 
            (projectInput () + (createProjectionParam(cellDim) * stabilizedPrevOutput)) + 
            (createDiagWeightParam(cellDim) .* Fun (Stabilize<'ElementType>((ct.toVar), device)))
            |> Tensor.sigmoid
        
        let ht : Tensor = (ot .* ct |> Tensor.tanh)

        let c : Tensor = ct
        let h : Tensor = 
            if (outputDim <> cellDim) 
            then (createProjectionParam(outputDim) * Fun(Stabilize<'ElementType>(ht.toVar, device)))
            else ht

        (h |> Tensor.toFunction, c |> Tensor.toFunction)

let LSTMPComponentWithSelfStabilization<'ElementType>(
    input:Variable,
    outputShape:NDShape, 
    cellShape:NDShape,
    recurrenceHookH:Variable -> Function,
    recurrenceHookC:Variable -> Function,
    device:DeviceDescriptor) : (Function * Function) =

        let dh = Variable.PlaceholderVariable(outputShape, input.DynamicAxes)
        let dc = Variable.PlaceholderVariable(cellShape, input.DynamicAxes)

        let LSTMCell = LSTMPCellWithSelfStabilization<'ElementType>(input, dh, dc, device)
        let actualDh = recurrenceHookH (new Variable(fst LSTMCell))
        let actualDc = recurrenceHookC (new Variable(snd LSTMCell))

        // TODO check this, seems to involve some mutation
        // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
        let replacement : IDictionary<Variable, Variable> =
            [
                dh, new Variable(actualDh)
                dc, new Variable(actualDc)
            ]
            |> dict

        (fst LSTMCell).ReplacePlaceholders(replacement)

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
let LSTMSequenceClassifierNet(
    input:Variable, 
    numOutputClasses:int, 
    embeddingDim:int, 
    LSTMDim:int, 
    cellDim:int, 
    device:DeviceDescriptor, 
    outputName:string) =

        let embeddingFunction : Function = Embedding(input, embeddingDim, device)
        let pastValueRecurrenceHook : (Variable -> Function) = fun x -> CNTKLib.PastValue(x)
        let (LSTMFunction:Function), _ = 
            LSTMPComponentWithSelfStabilization<single>(
                new Variable(embeddingFunction),
                shape [ LSTMDim ],
                shape [ cellDim ],
                pastValueRecurrenceHook,
                pastValueRecurrenceHook,
                device)

        let thoughtVectorFunction : Function = CNTKLib.SequenceLast(new Variable(LSTMFunction))

        FullyConnectedLinearLayer(new Variable(thoughtVectorFunction), numOutputClasses, device, outputName)

let DataFolder = __SOURCE_DIRECTORY__

let Train (device:DeviceDescriptor) =

    let inputDim = 2000
    let cellDim = 25
    let hiddenDim = 25
    let embeddingDim = 50
    let numOutputClasses = 5

    // build the model
    let featuresName = "features"
    let features = Variable.InputVariable(shape [ inputDim ], DataType.Float, featuresName, null, true)
    let labelsName = "labels"
    let labels = Variable.InputVariable(shape [ numOutputClasses ], DataType.Float, labelsName, ResizeArray<Axis>( [ Axis.DefaultBatchAxis() ]), true)

    let classifierOutput = LSTMSequenceClassifierNet(features, numOutputClasses, embeddingDim, hiddenDim, cellDim, device, "classifierOutput")
    let trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(classifierOutput), labels, "lossFunction")
    let prediction = CNTKLib.ClassificationError(new Variable(classifierOutput), labels, "classificationError")

    // prepare training data
    let streamConfigurations = 
        [|
            new StreamConfiguration(featuresName, inputDim, true, "x")
            new StreamConfiguration(labelsName, numOutputClasses, false, "y")
        |]

    let minibatchSource = 
        MinibatchSource.TextFormatMinibatchSource(
            Path.Combine(DataFolder, "Train.ctf"), 
            streamConfigurations, 
            MinibatchSource.InfinitelyRepeat, 
            true
            )

    let featureStreamInfo = minibatchSource.StreamInfo(featuresName)
    let labelStreamInfo = minibatchSource.StreamInfo(labelsName)

    // prepare for training
    let learningRatePerSample = new TrainingParameterScheduleDouble(0.0005, uint32 1)

    let momentumTimeConstant = CNTKLib.MomentumAsTimeConstantSchedule(256.)

    let parameterLearners = 
        ResizeArray<Learner>(
            [
                Learner.MomentumSGDLearner(
                    classifierOutput.Parameters(),
                    learningRatePerSample,
                    momentumTimeConstant,
                    true) // unitGainMomentum
            ]
            )

    let trainer = Trainer.CreateTrainer(classifierOutput, trainingLoss, prediction, parameterLearners)

    // train the model
    let minibatchSize = uint32 200
    let outputFrequencyInMinibatches = 20
    let miniBatchCount = 0
    let numEpochs = 5

    let report = progress (trainer, outputFrequencyInMinibatches) 

    let rec learnEpoch (step,epoch) = 

        if epoch <= 0
        // we are done
        then ignore ()
        else
            let step = step + 1
            let minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device)

            let arguments =
                [
                    features, minibatchData.[featureStreamInfo]
                    labels, minibatchData.[labelStreamInfo]
                ]
                |> dict

            trainer.TrainMinibatch(arguments, device) |> ignore

            report step |> printer
            
            // MinibatchSource is created with MinibatchSource.InfinitelyRepeat.
            // Batching will not end. Each time minibatchSource completes an sweep (epoch),
            // the last minibatch data will be marked as end of a sweep. We use this flag
            // to count number of epochs.
            let epoch = 
                if (MiniBatchDataIsSweepEnd(minibatchData.Values))
                then epoch - 1
                else epoch

            learnEpoch (step,epoch)

    learnEpoch (0,numEpochs)

let device = DeviceDescriptor.CPUDevice

Train device
