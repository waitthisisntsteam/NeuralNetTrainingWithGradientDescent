using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetTrainingGradientDescent
{
    public class NeuralNetwork
    {
        public Layer[] Layers;
        public ErrorFunction Error;

        public NeuralNetwork(ActivationFunction[] activation, ErrorFunction error, params int[] neuronsPerLayer)
        {
            Error = error;
            Layers = new Layer[neuronsPerLayer.Length];

            Layer inputLayer = new Layer(activation[0], error, neuronsPerLayer[0], null);
            Layer previousLayer = inputLayer;
            Layers[0] = inputLayer;
            for (int i = 1; i < neuronsPerLayer.Length; i++)
            {
                Layer currentLayer = new Layer(activation[0], error, neuronsPerLayer[i], previousLayer);
                previousLayer = currentLayer;
                Layers[i] = currentLayer;
            }
        }

        public void Randomize(Random random, double min, double max)
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].Randomize(random, min, max);
            }
        }

        public double[] Compute(double[] inputs)
        {
            for (int i = 0; i < Layers[0].Neurons.Length; i++)
            {
                Layers[0].Neurons[i].Output = inputs[i];
            }
            for (int i = 1; i < Layers.Length - 1; i++)
            {
                Layers[i].Compute();
            }

            return Layers[Layers.Length - 1].Compute();
        }

        public double GetError(double[] inputs, double[] desiredOutputs)
        {
            double errorSum = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                errorSum += Error.FunctionFunc(inputs[i], desiredOutputs[i]);
            }

            return errorSum;
        }

        public void ApplyUpdates()
        {
            for (int i = 0; i < Layers.Length; i++)
            {
                Layers[i].ApplyUpdates();
            }
        }

        public void Backprop(double learningRate, double[] desiredOutputs)
        {
            for (int i = Layers.Length - 1; i > 0; i--)
            {
                Layers[i].Backprop(learningRate);
            }
        }

        public double Train(double[][] inputs, double[][] desiredOutputs, double learingRate)
        {
            return 0;
        }
    }
}
