import Header from "../components/Header";
import "../components/Styles/params.css";
import { GlobalContext } from "../context";
import { useContext } from "react";

export default function Params() {
  console.log(GlobalContext);
  const { payment, setPayment, time, setTime, profit, setProfit } =
    useContext(GlobalContext);
  return (
    <div className="params-page">
      <Header />
      <div className="form-container">
        <div className="input-group">
          <label htmlFor="payment">Payment</label>
          <input
            type="number"
            id="payment"
            name="payment"
            placeholder="Enter Payment"
            value={payment}
            onChange={(e) => setPayment(e.target.value)}
          />
        </div>
        <div className="input-group">
          <label htmlFor="time">Time</label>
          <input
            type="number"
            id="time"
            name="time"
            placeholder="Enter Time"
            value={time}
            onChange={(e) => setTime(e.target.value)}
          />
        </div>
        <div className="input-group">
          <label htmlFor="profit">Profit</label>
          <input
            type="number"
            id="profit"
            name="profit"
            placeholder="Enter Profit"
            value={profit}
            onChange={(e) => setProfit(e.target.value)}
          />
        </div>
      </div>
    </div>
  );
}
