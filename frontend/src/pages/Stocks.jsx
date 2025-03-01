import { useEffect, useContext, useState } from "react";
import Header from "../components/Header";
import { GlobalContext } from "../context";
import StockBody from "../components/StockBody";
import axios from "axios";
import LoadingSpinner from "../components/LoadingSpinner";
import FinanceBackground from "../components/FinanceBackground";
import "../components/Styles/stocks.css";
export default function Stocks() {
  let dataArr = createDefaultData();
  const { payment, time, profit } = useContext(GlobalContext);
  const [loading, setLoading] = useState(false);

  async function handleGetStocks() {
    try {
      setLoading(true);
      const response = await axios.get("http://localhost:33306/", {
        params: {
          investment_amount: payment,
          time_horizon: time,
          expected_return: profit,
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
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    handleGetStocks();
  }, []);

  return (
    <div className="stocks-main">
      {/* SwirlAnimation as background */}
      <div className="background-container">
        <FinanceBackground />
      </div>

      {/* Page content */}
      <Header backButton={true} />
      {!loading && <StockBody dataArr={dataArr} />}
      {loading && <LoadingSpinner />}
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
  const data5 = {
    name: "amazon",
    price: 3344,
    risk: 0.55,
    profit: 230,
    code: "AMZN",
  };
  const data6 = {
    name: "microsoft",
    price: 299,
    risk: 0.48,
    profit: 100,
    code: "MSFT",
  };
  const data7 = {
    name: "facebook",
    price: 375,
    risk: 0.6,
    profit: 80,
    code: "META",
  };
  const data8 = {
    name: "nvidia",
    price: 850,
    risk: 0.2,
    profit: 150,
    code: "NVDA",
  };
  const data9 = {
    name: "netflix",
    price: 650,
    risk: 0.5,
    profit: 120,
    code: "NFLX",
  };
  const data10 = {
    name: "paypal",
    price: 270,
    risk: 0.4,
    profit: 60,
    code: "PYPL",
  };

  return [
    data1,
    data2,
    data3,
    data4,
    data5,
    data6,
    data7,
    data8,
    data9,
    data10,
  ];
}
