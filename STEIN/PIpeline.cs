using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace STEIN
{
    /// <summary>
    /// A sequential execution of stages for convenience.
    /// </summary>
    public class Pipeline : IPipelineStage
    {
        public List<IPipelineStage> Stages
        { get; set; }

        public Dictionary<IPipelineStage, object> Results
        { get; set; }

        public Pipeline()
        {
            Stages = new List<IPipelineStage>();
            Results = new Dictionary<IPipelineStage, object>();
        }


        public object Run(object input)
        {
            object lastResult = input;
            foreach(var stage in Stages)
            {
                lastResult = stage.Run(lastResult);
                Results[stage] = lastResult;
            }

            return lastResult;
        }
    }
}
