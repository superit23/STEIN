using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace STEIN
{
    /// <summary>
    /// Represents an algorithm that iteratively finds a better set of solutions.
    /// </summary>
    public interface IOptimizer
    {
        Func<Model, float> ObjectiveFunction
        { get; set; }

        float[] Run();
    }
}
