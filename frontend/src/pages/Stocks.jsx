import Header from "../components/Header";
import StockBody from "../components/StockBody";

export default function Stocks() {
  const dataArr = createDefaultData();

  return (
    <div className="stocks-main">
      <Header backButton={true} />
      <StockBody dataArr={dataArr} />
    </div>
  );
}

function createDefaultData() {
  const data1 = {
    name: "stock",
    price: 3,
    risk: 0.43,
    profit: 132,
    code: "GOOGL",
  };
  const data2 = {
    name: "otherStock",
    price: 43,
    risk: 0.32,
    profit: 0,
    code: "NA",
  };
  const data3 = {
    name: "tesla",
    price: 1000,
    risk: 0.99,
    profit: 54,
    code: "LI",
  };
  const data4 = {
    name: "apple",
    price: 41315,
    risk: 0.4,
    profit: 14313,
    code: "TU",
  };
  return [data1, data2, data3, data4];
}
