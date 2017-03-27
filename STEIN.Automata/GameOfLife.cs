using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace STEIN.Automata
{
    /// <summary>
    /// The <see cref="GameOfLife"/> is a Turing complete model of celluluar automata.
    /// </summary>
    /// Adapted from https://gist.github.com/joelmartinez/1194013
    public class GameOfLife
    {
        private bool[,] world;
        private bool[,] nextGeneration;
        private Task processTask;

        /// <summary>
        /// Instantiate a <see cref="GameOfLife"/> on a square grid of <paramref name="size"/>.
        /// </summary>
        /// <param name="size">Length of square grid's sides.</param>
        public GameOfLife(int size)
        {
            if (size < 0) throw new ArgumentOutOfRangeException("Size must be greater than zero");
            Size = size;
            world = new bool[size, size];
            nextGeneration = new bool[size, size];
        }


        public int Size { get; private set; }
        public int Generation { get; private set; }

        public delegate void GenerationComplete(GameOfLife sim);
        public event GenerationComplete GenerationCompleted;

        public bool this[int x, int y]
        {
            get { return world[x, y]; }
            set { world[x, y] = value; }
        }


        /// <summary>
        /// Run the next generation of the <see cref="GameOfLife"/>.
        /// </summary>
        /// <returns></returns>
        public Task RunGeneration()
        {
            if (processTask != null)
            {
                if (processTask.IsCompleted)
                {
                    // When a generation has completed
                    // Now flip the back buffer, so we can start processing on the next generation
                    var flip = nextGeneration;
                    nextGeneration = world;
                    world = flip;
                    Generation++;
                }
                else
                {
                    throw new InvalidOperationException("New generation cannot be started before the previous has completed.");
                }
            }

            // Begin the next generation's processing asynchronously
            processTask = ProcessGeneration();
            return processTask;
        }


        private Task ProcessGeneration()
        {
            return Task.Factory.StartNew(() =>
            {
                for(int x = 0; x < Size; x++)
                {
                    Parallel.For(0, Size, y =>
                    {
                        int numberOfNeighbors = IsNeighborAlive(x, y, -1, 0)
                            + IsNeighborAlive(x, y, -1, 1)
                            + IsNeighborAlive(x, y, 0, 1)
                            + IsNeighborAlive(x, y, 1, 1)
                            + IsNeighborAlive(x, y, 1, 0)
                            + IsNeighborAlive(x, y, 1, -1)
                            + IsNeighborAlive(x, y, 0, -1)
                            + IsNeighborAlive(x, y, -1, -1);

                        bool isAlive = world[x, y];

                        nextGeneration[x, y] = (isAlive && (numberOfNeighbors == 2 || numberOfNeighbors == 3)) ||
                        !isAlive && numberOfNeighbors == 3;
                    });
                }

                if (GenerationCompleted != null) GenerationCompleted(this);
            });
        }


        private int IsNeighborAlive(int x, int y, int offsetx, int offsety)
        {
            int result = 0;

            int proposedOffsetX = x + offsetx;
            int proposedOffsetY = y + offsety;
            bool outOfBounds = proposedOffsetX < 0 || proposedOffsetX >= Size | proposedOffsetY < 0 || proposedOffsetY >= Size;

            if (!outOfBounds)
            {
                result = world[x + offsetx, y + offsety] ? 1 : 0;
            }
            return result;
        }
    }

}
