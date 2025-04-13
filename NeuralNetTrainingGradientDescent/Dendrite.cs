using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetTrainingGradientDescent
{
    public class Dendrite
    {
        public Neuron Next { get; set; }
        public Neuron Previous { get; set; }
        public double Weight { get; set; }

        public double WeightUpdate { get; set; }

        public Dendrite(Neuron next, Neuron previous, double weight) => (Next, Previous, Weight) = (next, previous, weight);

        public double Compute() 
        {
            return Previous.Output * Weight;
        }

        public void ApplyUpdates()
        {
            Weight += WeightUpdate;
            WeightUpdate = 0;
        }
    }
}
