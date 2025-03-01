import Header from "../components/Header";
import "../components/Styles/params.css";
import FormContainer from "../components/FormContainer";
import { GlobalContext } from "../context";
import StocksBtn from "../components/StocksBtn";
import RotatingCube from "../components/CubeBackground";

export default function Params() {
  return (
    <div className="params-page">
      <RotatingCube />
      <Header />
      <FormContainer />
    </div>
  );
}
