namespace CNTK.FSharp

[<AutoOpen>]
module Readers = 

    open CNTK

    type StreamConfig = unit -> StreamConfiguration

    type DataSource = {
        SourcePath: string
        Streams: StreamConfig seq
        }

    type Stream () =
        static member config (name:string,dim:int) = 
            fun () -> new StreamConfiguration (name,dim)
        static member config (name:string,dim:int,sparse:bool) = 
            fun () -> new StreamConfiguration (name,dim,sparse)
        static member config (name:string,dim:int,sparse:bool,alias:string) = 
            fun () -> new StreamConfiguration (name,dim,sparse,alias)
        
    type StreamProcessing = 
        | FullDataSweep
        | InfinitelyRepeat

    let textSource (data:DataSource) =
        let streams = 
            data.Streams
            |> Seq.map (fun f -> f ()) 
            |> ResizeArray
        fun (processing: StreamProcessing) ->
            MinibatchSource.TextFormatMinibatchSource(
                data.SourcePath, 
                streams, 
                match processing with
                | FullDataSweep -> MinibatchSource.FullDataSweep
                | InfinitelyRepeat -> MinibatchSource.InfinitelyRepeat)