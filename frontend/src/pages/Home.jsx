import Header from "../components/Header";
import Body from "../components/Body";
import ParamsButton from "../components/ParamsBtn";
import "../components/Styles/body.css";
export default function Home() {
  return (
    <div className="home-container">
      <Header />
      <Body />
      <ParamsButton route="/params" />
    </div>
  );
}
