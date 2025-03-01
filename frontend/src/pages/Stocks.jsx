import { useEffect, useContext } from "react";
import Header from "../components/Header";
import { GlobalContext } from "../context";
import StockBody from "../components/StockBody";
import axios from "axios";
export default function Stocks() {
  let dataArr = createDefaultData();
  const { payment, time, profit } = useContext(GlobalContext);

  async function handleGetStocks() {
    try {
      const response = await axios.get("http://localhost:33306/", {
        params: {
          payment: payment,
          time: time,
          profit: profit,
        },
      });
      const result = await response.data;

      if (result) {
        dataArr = result;
      } else {
        dataArr = createDefaultData();
      }
    } catch (e) {
      console.log("data not fetched " + e);
      dataArr = createDefaultData();
    }
  }
  useEffect(() => {
    handleGetStocks();
  }, []);

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
