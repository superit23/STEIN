using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Automata
{
    public class Transition<TState, TSymbol>
    {
        public TState StartState { get; private set; }
        public TSymbol Symbol { get; private set; }
        public TState EndState { get; private set; }

        public Transition(TState startState, TSymbol symbol, TState endState)
        {
            StartState = startState;
            Symbol = symbol;
            EndState = endState;
        }

        public override string ToString()
        {
            return string.Format("({0}, {1}) -> {2}", StartState, Symbol, EndState);
        }
    }
}
