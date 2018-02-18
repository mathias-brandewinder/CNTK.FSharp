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

    /// Map one variable name to the corresponding name
    /// in the data file.
    type NameMapping = {
        VariableName:string
        SourceName:string
        }

    /// Map all Variables to their corresponding names
    /// in the data file.
    type NameMappings = {
        Features : NameMapping seq
        Labels : NameMapping
        }

    let extractMappings (desc:NameMappings) (model:Function) =
        
        let extractFeature name = 
            model.Inputs
            |> Seq.filter (fun i -> i.Name = name)
            |> Seq.exactlyOne
        let extractLabel name = 
            model.Outputs
            |> Seq.filter (fun i -> i.Name = name)
            |> Seq.exactlyOne
        
        let mappings : InputMappings = {
            Features = 
                desc.Features
                |> Seq.map (fun mapping -> 
                    { 
                        Variable = extractFeature mapping.VariableName
                        SourceName = mapping.SourceName } 
                    )
                |> Seq.toList
            Labels =
                { 
                    Variable = extractLabel desc.Labels.VariableName
                    SourceName = desc.Labels.SourceName
                }
            }

        mappings