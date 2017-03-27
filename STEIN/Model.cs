using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Reflection;

namespace STEIN
{
    /// <summary>
    /// A representation of a system that for some input returns a corresponding output.
    /// Specifically, an instance of a <see cref="Model"/> contains the algorithmic properties
    /// a well as parameters and hyperparameters that encode the behavior of a system.
    /// </summary>
    public abstract class Model : IPipelineStage
    {
        public IEnumerable<Hyperparameter> Hyperparameters
        {
            get
            {
                return GetType().CustomAttributes.OfType<Hyperparameter>();
            }
        }

        public abstract object Run(object input);
    }
}
