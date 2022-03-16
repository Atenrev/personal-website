import * as React from "react"
import { FaTerminal } from "@react-icons/all-files/fa/FaTerminal";
import { FaGithub } from "@react-icons/all-files/fa/FaGithub";
import { FaLinkedin } from "@react-icons/all-files/fa/FaLinkedin";
import { FaPowerOff } from "@react-icons/all-files/fa/FaPowerOff";
import { FaVolumeUp } from "@react-icons/all-files/fa/FaVolumeUp";
import { FaWifi } from "@react-icons/all-files/fa/FaWifi";
import { StaticImage } from "gatsby-plugin-image"

import "../styles/index.scss"
import Seo from "../components/Seo"
import Prompt from "../components/Prompt"


// markup
const IndexPage = () => {
  let [currentDate, setCurrentDate] = React.useState(new Date().toLocaleTimeString());

  let updateDate = () => {
    setCurrentDate(new Date().toLocaleTimeString());
  }

  React.useEffect(() => {
    let interval = setInterval(updateDate, 1000);

    return function cleanup() {
      clearInterval(interval);
    };
  }, []);

  return (
    <main>
      <Seo
        title="Home"
        keywords={['sergi', 'masip', 'sergi masip',
          'software', 'data', 'web developer', 'app developer',
          'engineer', 'developer', 'machine learning', 'artifical intelligence',
          'unity3D', 'pytorch', 'computer vision', 'deep learning', 'research',
          'uab', 'cvc', 'computer vision center', 'universitat autonoma de catalunya']}
      />
      <div className="background">
        <StaticImage
          src="../images/background.jpg"
          alt="Background image of a mountain."
          placeholder="blurred"
          layout="fixed"
          style={{ height: "100%", width: "100%" }}
        />
      </div>
      <div className="os-container">

        <header className="panel">
          <div className="column">
            <div className="icon active">
              <FaTerminal />
            </div>
            <div className="icon hoverable">
              <a href="https://github.com/Atenrev">
                <FaGithub />
              </a>
            </div>
            <div className="icon hoverable">
              <a href="https://www.linkedin.com/in/sergimasipcabeza/">
                <FaLinkedin />
              </a>
            </div>
          </div>
          <div className="column center-row">
            <div>{currentDate}</div>
          </div>
          <div className="column right-row">
            <div className="icon">
              <FaWifi />
            </div>
            <div className="icon">
              <a className="invisible-link" href="https://www.youtube.com/watch?v=dQw4w9WgXcQ">
                <FaVolumeUp />
              </a>
            </div>
            <div className="icon">
              <FaPowerOff />
            </div>
          </div>
        </header>

        <div className="windows-container">
          <div className="panel" style={{ gridArea: "about" }}>
            <Prompt command="head">about_me.txt</Prompt>
            <p>I am a Spanish Computer Engineering student focused on Artificial Intelligence and Machine Learning with wide experience in web, application, and video game development.</p>
            <Prompt command="ls -l">experience/</Prompt>
            <p>
              -rw-r--r-- 1 sergi masip 4096 Jun 31 20:22 <a href="https://www.uab.cat/">uab-computer-engineering-graduation.txt</a><br />
              -rw-r--r-- 1 sergi masip 341&nbsp; Dec 31 20:21 <a href="http://www.cvc.uab.es/">4-month-CVC-research-internship.txt</a><br />
              -rw-r--r-- 1 sergi masip 4096 Dec 31 20:20 <a href="http://humaninputs.com/">human-inputs-indie-video-game-developer.txt</a><br />
              -rw-r--r-- 1 sergi masip 256&nbsp; Sep 15 20:19 <a href="http://eqtic.net/">3-month-exp-EQTic-as-webdev.txt</a><br />
              -rw-r--r-- 1 sergi masip 1024 Sep 15 20:18 <a href="http://eqtic.net/">1-year-exp-EQTic-as-webdev.txt</a><br />
              -rw-r--r-- 1 sergi masip 2048 Jan 31 20:18 <a href="https://copernic.cat/">hnc-cross-platform-app-dev-graduation.txt</a><br />
            </p>
            <Prompt command="ls">skills/</Prompt>
            <p style={{ display: "flex", flexDirection: "row", flexWrap: "wrap", gap: "24px" }}>
              <span>Machine-Learning</span>
              <span>Artificial-Intelligence</span>
              <span>Computer-Vision</span>
              <span>Pytorch</span>
              <span>OpenCV</span>
              <span>Tensorflow/keras</span>
              <span>Sklearn</span>
              <span>Unity 3D</span>
              <span>Django</span>
              <span>React</span>
              <span>Flutter</span>
            </p>
            <Prompt command="tail">./mywebsite/footer.html</Prompt>
            <p>Copyright © Sergi Masip 2022 All rights reserved</p>
          </div>

          <div className="panel" style={{ gridArea: "projects" }}>
            <Prompt command="tree">my_projects/</Prompt>
            <p>
              my-projects/<br />
              ├─ human_inputs/<br />
              │  ├─ <a href="https://play.google.com/store/apps/details?id=com.HUMANINPUTS.taptapflamingo">tap_tap_flamingo.apk</a><br />
              │  ├─ <a href="https://play.google.com/store/apps/details?id=com.HUMANINPUTS.FINGERIT">fingerit.apk</a><br />
              │  ├─ <a href="https://humaninputs.itch.io/hackterm">hackterm.exe</a><br />
              ├─ <a href="https://github.com/Atenrev/Deep-Image-Steganography-Reimplementation">Deep-Image-Steganography-Reimplementation.py</a><br />
              ├─ <a href="https://github.com/Atenrev/forocoches-language-generation">forocoches-language-generation.py</a>
              ├─ <a href="https://github.com/Atenrev/chessformers">chessformers.py</a>
            </p>
          </div>

          <div className="panel" style={{ gridArea: "other" }}>
            <Prompt>./contact.me</Prompt>
            <p>Send an email to <a href="mailto:hello@sergimasip.com">hello@sergimasip.com</a></p>
            <p>Check my awesome projects at <a href="https://github.com/Atenrev">Github</a></p>
            <p>Add me on <a href="https://www.linkedin.com/in/sergimasipcabeza/">Linkedin</a></p>
            <Prompt command=""><span className="blink-animation">▮</span></Prompt>
          </div>
        </div>
      </div>
    </main>
  )
}

export default IndexPage
