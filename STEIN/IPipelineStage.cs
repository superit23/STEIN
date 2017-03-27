using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace STEIN
{
    public interface IPipelineStage
    {
        object Run(object input);
    }
}
