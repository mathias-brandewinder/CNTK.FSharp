namespace CNTK.FSharp

[<RequireQualifiedAccess>]
module TextFormat =

    open CNTK

    /// Map a Variable to its name in the data file.
    type InputMapping = {
        Variable:Variable
        SourceName:string
        }

    /// Map all model variables to their corresponding name
    /// in the data file.
    type InputMappings = {
        Features: InputMapping seq
        Labels: InputMapping
        } with
        member this.Mappings =
            seq {
                yield! this.Features
                yield this.Labels
                }

    let streams (mappings:InputMappings) =
        let size (v:Variable) = 
            v.Shape.Dimensions 
            |> Seq.fold (*) 1
        mappings.Mappings
        |> Seq.map (fun mapping ->
            new StreamConfiguration(
                mapping.SourceName, 
                mapping.Variable |> size |> uint32,
                mapping.Variable.IsSparse)
            )
        |> ResizeArray

    /// Create a MinibatchSource from a file.
    let source (filePath:string, mappings:InputMappings) =
        let streams = streams mappings
        // TODO: include read strategy,
        // ex: full sweep vs batch size, randomization, ...
        MinibatchSource.TextFormatMinibatchSource(
            filePath,
            streams,
            MinibatchSource.InfinitelyRepeat)         
    