using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace STEIN
{
    public class Hyperparameter : Attribute
    {
        public Type Type
        { get; set; }

        // Can't include Actions in Attributes
        //public Action<Boolean> Constraints
        //{ get; set; }


    }
}
