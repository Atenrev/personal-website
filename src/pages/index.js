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
import { Link } from "gatsby"


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
              <Link to="https://github.com/Atenrev">
                <FaGithub />
              </Link>
            </div>
            <div className="icon hoverable">
              <Link to="https://www.linkedin.com/in/sergimasipcabeza/">
                <FaLinkedin />
              </Link>
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
              <Link className="invisible-link" to="https://www.youtube.com/watch?v=dQw4w9WgXcQ">
                <FaVolumeUp />
              </Link>
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
              -rw-r--r-- 1 sergi masip 4096 Jun 31 20:22 <Link to="https://www.uab.cat/">uab_computer_engineering_graduation.txt</Link><br />
              -rw-r--r-- 1 sergi masip 341&nbsp; Dec 31 20:21 <Link to="http://www.cvc.uab.es/">4_month_CVC_research_internship.txt</Link><br />
              -rw-r--r-- 1 sergi masip 256&nbsp; Sep 15 20:19 <Link to="https://eqtic.net/">3_month_exp_EQTic_as_webdev.txt</Link><br />
              -rw-r--r-- 1 sergi masip 1024 Sep 15 20:18 <Link to="https://eqtic.net/">1_year_exp_EQTic_as_webdev.txt</Link><br />
              -rw-r--r-- 1 sergi masip 2048 Jan 31 20:18 <Link to="https://copernic.cat/">hnc_cross_platform_app_dev_graduation.txt</Link><br />
            </p>
            <Prompt command="ls">skills/</Prompt>
            <p>
              <div style={{ display: "flex", flexDirection: "row", flexWrap: "wrap", gap: "24px" }}>
                <div>Machine Learning</div>
                <div>Artificial Intelligence</div>
                <div>Computer Vision</div>
                <div>Pytorch</div>
                <div>OpenCV</div>
                <div>Tensorflow/keras</div>
                <div>Sklearn</div>
                <div>Django</div>
                <div>Unity 3D</div>
                <div>Flutter</div>
              </div>
            </p>
            <Prompt command="tail">./mywebsite/footer.html</Prompt>
            <p>Copyright © Sergi Masip 2022 All rights reserved</p>
          </div>

          <div className="panel" style={{ gridArea: "projects" }}>
            <Prompt command="tree">my_projects/</Prompt>
            <p>
              my-projects/<br />
              ├─ human_inputs/<br />
              │  ├─ <Link to="https://play.google.com/store/apps/details?id=com.HUMANINPUTS.taptapflamingo">tap_tap_flamingo.apk</Link><br />
              │  ├─ <Link to="https://play.google.com/store/apps/details?id=com.HUMANINPUTS.FINGERIT">fingerit.apk</Link><br />
              │  ├─ <Link to="https://humaninputs.itch.io/hackterm">hackterm.exe</Link><br />
              ├─ <Link to="https://github.com/Atenrev/Deep-Image-Steganography-Reimplementation">Deep-Image-Steganography-Reimplementation.py</Link>
            </p>
          </div>

          <div className="panel" style={{ gridArea: "other" }}>
            <Prompt>./contact.me</Prompt>
            <p>Send an email to <a href="mailto:hello@sergimasip.com">hello@sergimasip.com</a></p>
            <p>Check my awesome projects at <Link to="https://github.com/Atenrev">Github</Link></p>
            <p>Add me on <Link to="https://www.linkedin.com/in/sergimasipcabeza/">Linkedin</Link></p>
            <Prompt><span className="blink-animation">▮</span></Prompt>
          </div>
        </div>
      </div>
    </main>
  )
}

export default IndexPage
