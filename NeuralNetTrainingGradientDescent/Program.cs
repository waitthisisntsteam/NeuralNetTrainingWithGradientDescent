using System;

namespace NeuralNetTrainingGradientDescent
{
    internal class Program
    {
        static void Main(string[] args)
        {
            ActivationErorrFormulas activationErorrFormulas = new ActivationErorrFormulas();
            ErrorFunction Error = new ErrorFunction(activationErorrFormulas.MeanSquared, activationErorrFormulas.MeanSquaredD);
            ActivationFunction Activation = new ActivationFunction(activationErorrFormulas.Sigmoid, activationErorrFormulas.Sigmoid);

            NeuralNetwork network = new NeuralNetwork([Activation], Error, 1, 5, 5, 5, 5, 1);
            network.Randomize(new Random(), 0.25, 0.75);

            const int number = 16;
            double[][] desiredOutputs = new double[number][];
            double[][] inputs = new double[number][];

            for (int i = 0; i < number; i++)
            {
                inputs[i] = [(Math.PI / number) * i];

                desiredOutputs[i] = [Math.Sin(inputs[i][0])];
            }

            double error = network.Train(inputs, desiredOutputs, 0.0005);
            double originalError = error;
            double oldError = originalError;
            while (true)
            {
                Console.WriteLine("Starting Error:");
                Console.WriteLine(originalError);
                Console.WriteLine("      ");

                Console.WriteLine("Current Error:      Current Outputs:");
                error = network.Train(inputs, desiredOutputs, 0.0005);
                Console.Write(error);

                Console.Write("                   ");
                for (int i = 0; i < inputs.Length; i++)
                {
                    var output = network.Compute(inputs[i]);
                    for (int j = 0; j < output.Length; j++) Console.Write(output[j] + " ");
                }

                if (oldError < error)
                {
                    ;//you goofed
                }
                oldError = error;
                //Console.ReadKey();
                Console.Clear();
            }
        }
    }
}