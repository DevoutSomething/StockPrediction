import React from "react";
import Header from "../components/Header";
import Body from "../components/Body";
import ParamsButton from "../components/ParamsBtn";
import RollingHills from "../components/RollingHills";
import "../components/Styles/body.css";

export default function Home() {
  return (
    <div className="home-container">
      <RollingHills />
      <Header />
      <Body />
      <ParamsButton route="/params" />
    </div>
  );
}
