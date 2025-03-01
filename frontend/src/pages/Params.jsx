import Header from "../components/Header";
import "../components/Styles/params.css";
import FormContainer from "../components/FormContainer";
import { GlobalContext } from "../context";
import StocksBtn from "../components/StocksBtn";
export default function Params() {
  return (
    <div className="params-page">
      <Header />
      <FormContainer />
    </div>
  );
}
