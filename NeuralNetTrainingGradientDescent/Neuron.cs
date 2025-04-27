using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Transactions;

namespace NeuralNetTrainingGradientDescent
{
    public class Neuron
    {
        public double Bias;
        public Dendrite[] Dendrites;

        public double Output { get; set; }
        public double Input { get; private set; }
        public ActivationFunction Activation { get; set; }
        public ErrorFunction Error { get; set; }

        public double Delta { get; set; }
        public double BiasUpdate;
        public double PreviousBiasUpdate;

        public Neuron(ActivationFunction activation, Neuron?[] previousNeurons)
        {
            Activation = activation;

            if (previousNeurons == null)
            {
                Dendrites = new Dendrite[0];
            }
            else
            {
                Dendrites = new Dendrite[previousNeurons.Length];
                for (int i = 0; i < previousNeurons.Length; i++)
                {
                    Dendrites[i] = new Dendrite(null, previousNeurons[i], 0);
                    for (int j = 0; j < previousNeurons[i].Dendrites.Length; j++)
                    {
                        previousNeurons[i].Dendrites[j].Next = this;
                    }
                }
            }
        }

        public void Randomize(Random random, double min, double max)
        {
            for (int i = 0; i < Dendrites.Length; i++)
            {
                Dendrites[i].Weight = (random.NextDouble() * (max - min)) + min;
            }
            Bias = (random.NextDouble() * (max - min)) + min;
        }

        public double Compute()
        {
            Input = Bias;
            for (int i = 0; i < Dendrites.Length; i++)
            {
                Input += Dendrites[i].Compute();
            }
            Output = Activation.FunctionFunc(Input);
            return Output;

        }
        public void ApplyUpdates(double momentum)
        {
            BiasUpdate += PreviousBiasUpdate * momentum;
            Bias += BiasUpdate;
            PreviousBiasUpdate = BiasUpdate;
            BiasUpdate = 0;

            for (int i = 0; i < Dendrites.Length; i++)
            {
                Dendrites[i].ApplyUpdates(momentum);
            }
        }

        public void Backprop(double learningRate)
        {
            double aPrimeZ = Activation.DerivativeFunc(Input);

            for (int i = 0; i < Dendrites.Length; i++)
            {
                Dendrites[i].Previous.Delta += Delta * aPrimeZ * Dendrites[i].Weight;

                double weightPartialDerivative = Delta * aPrimeZ * Dendrites[i].Previous.Output;
                Dendrites[i].WeightUpdate += learningRate * -weightPartialDerivative;
            }

            double biasPartialDerivative = Delta * aPrimeZ;
            BiasUpdate += learningRate * -biasPartialDerivative;

            Delta = 0;
        }
    }
}