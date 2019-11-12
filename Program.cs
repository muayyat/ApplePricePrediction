using System;
using System.IO;
using System.Linq;
using Microsoft.ML;
namespace ApplePricePrediction
{
    public class Program
    {
        static readonly string _trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "ApplePrice.zip");
        public static void Main(string[] args)
        {
            MLContext mLContext = new MLContext(seed: 0);
            var model = Train(mLContext, _trainingDataPath);
            Evaluate(mLContext, model);
            TestSinglePrediction(mLContext, model);
        }
      
        public static ITransformer Train(MLContext mLContext, string dataPath)
        {
            IDataView dataView = mLContext.Data.LoadFromTextFile<ApplePrice>(dataPath, hasHeader: true, separatorChar: ',');
            var pipeline = mLContext.Transforms.CopyColumns(outputColumnName: "Label", inputColumnName: "Price")
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "LocationEncoded", inputColumnName: "Location"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "ColorEncoded", inputColumnName: "Color"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "NumberEncoded", inputColumnName: "Number"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "SizeEncoded", inputColumnName: "Size"))
                .Append(mLContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: "PaymentType"))
                .Append(mLContext.Transforms.Concatenate("Features","LocationEncoded","ColorEncoded","NumberEncoded","SizeEncoded","PaymentTypeEncoded","Price"))
                .Append(mLContext.Regression.Trainers.FastTree())
                ;
            var model = pipeline.Fit(dataView);
            return model;
            
        }
        private static void Evaluate(MLContext mLContext, ITransformer model)
        {
            IDataView dataView = mLContext.Data.LoadFromTextFile<ApplePrice>(_testDataPath, hasHeader: true, separatorChar: ',');
            var predictions = model.Transform(dataView);
            var metrics = mLContext.Regression.Evaluate(predictions, "Label", "Score");
            Console.WriteLine();
            Console.WriteLine($"*   By Muayyat Billah  ");
            Console.WriteLine($"*------------------------");
            Console.WriteLine($"* RS Score: {metrics.RSquared:0.##}");
            Console.WriteLine($"* RMS Error: {metrics.RootMeanSquaredError:#.##}");

        }
        private static void TestSinglePrediction(MLContext mLContext, ITransformer model)
        {
            var predictionFunction = mLContext.Model.CreatePredictionEngine<ApplePrice, ApplePrediction>(model);
            var pricingSample = new ApplePrice()
            {
                Location = "New York",
                Color = "Yellow",
                Number = 1,
                Size = "Small",
                PaymentType = "Card",
                Price = 0 // To predict. Actual/Observed = 15.5
           

        };
            var prediction = predictionFunction.Predict(pricingSample);
            Console.WriteLine($"***********************");
            Console.WriteLine($"Number: {pricingSample.Number}");
            Console.WriteLine($"Color: {pricingSample.Color}");
            Console.WriteLine($"Predicted Price: {prediction.Price: 0.####}, Supposed Price:16.039");
            Console.WriteLine($"***********************");
            Console.WriteLine($"In Conclusion, {pricingSample.Number} {pricingSample.Size} Apple(s) will cost around {prediction.Price} in {pricingSample.Location} ");

        }
     ///   public static IWebHostBuilder CreateWebHostBuilder(string[] args) =>
       //     WebHost.CreateDefaultBuilder(args)
    //            .UseStartup<Startup>();
    }
}
