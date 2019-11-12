using Microsoft.ML.Data;

namespace ApplePricePrediction
{
    public class ApplePrice
    {
        [LoadColumn(0)]
        public string Location;

        [LoadColumn(1)]
        public string Color;

        [LoadColumn(2)]
        public int Number;

        [LoadColumn(3)]
        public string Size;

        [LoadColumn(4)]
        public string PaymentType;

        [LoadColumn(5)]
        public float Price;

    }
    public class ApplePrediction
    {
        [ColumnName("Score")]
        public float Price;
    }
}
