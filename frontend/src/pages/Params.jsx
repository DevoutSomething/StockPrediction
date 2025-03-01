import Header from "../components/Header";
import "../components/Styles/params.css";
import FormContainer from "../components/FormContainer";
import { useContext } from "react";
import { GlobalContext } from "../context";
import StocksBtn from "../components/StocksBtn";
export default function Params() {
  const { payment, time, profit } = useContext(GlobalContext);
  return (
    <div className="params-page">
      <Header />
      <FormContainer />
      {payment && time && profit && <StocksBtn route="/stocks" />}
    </div>
  );
}
