import React from "react"
import styled from "@emotion/styled"

import Layout from "../components/layout"
import SEO from "../components/seo"

const Separator = styled.div`
  clear: both;
`

const MarkedH3 = styled.h3`
  display: inline;
  border-radius: 1em 0 1em 0;
  background-color: rgba(255, 250, 150, 0.8);
`

const TimeLineItemContent = styled.div`
  
`

const AboutPage = () => (
    <Layout>
        <SEO
            title="About me"
            description="I am Sergi Masip, a programmer since the age of 14 and a computer engineer really interested in Artificial Intelligence and Machine Learning."
            keywords={['sergi', 'masip', 'sergi masip', 'about', 'uab', 'universitat', 'autonoma', 'barcelona', 'eqtic', 'eqt', 'nicolau copernic', 'student', 'computer engineer', 'videogame', 'unity', 'django', 'web developer']}
        />
        <div className="container">
            <h1>About me</h1>
            <p>
            I am Sergi Masip, a programmer since the age of 14 and a computer engineer really interested in Artificial Intelligence and Machine Learning. 
            I am also experienced in web development with Django and game development with Unity. 
            <br></br><br></br>
            Currently, I am working on a large application using Django as the backend and Flutter as the frontend. 
            Meanwhile, I am studying for a computer engineering degree at the <a href="https://www.uab.cat/">Autonomous University of Barcelona</a>.
            <br></br><br></br>I love Isaac Asimov.
            </p>
            <h2>Timeline of my experience</h2>
            <ul className="timeline">
                <TimeLineItem>
                    <MarkedH3>2020-</MarkedH3>
                    <p>I am currently working on a large application developed with Django that consists of a Dashboard and an API to which a mobile application built in Flutter connects.</p>
                </TimeLineItem>
                <TimeLineItem>
                    <MarkedH3>2019 Jun-Sep</MarkedH3>
                    <p>I worked for 3 months at <a href="https://eqtic.net/">EQTic</a>. I developed from scratch an internal application using Django. I also worked with Node-Red in an IoT-related web-app.</p>
                </TimeLineItem>
                <TimeLineItem>
                    <MarkedH3>2019 Jun</MarkedH3>
                    <p>I published <a href='https://humaninputs.itch.io/taptapflamingo'>Tap Tap Flamingo</a> with my indie videogame studio.</p>
                </TimeLineItem>
                <TimeLineItem>
                    <MarkedH3>2018 Dic</MarkedH3>
                    <p>I published <a href='https://humaninputs.itch.io/fingerit'>FINGERIT</a> with my indie videogame studio.</p>
                </TimeLineItem>
                <TimeLineItem>
                    <MarkedH3>2018 Nov</MarkedH3>
                    <p>I published <a href='https://humaninputs.itch.io/hackterm'>HACKTERM</a> with my indie videogame studio.</p>
                </TimeLineItem>
                <TimeLineItem>
                    <MarkedH3>2018-2022</MarkedH3>
                    <p>I started my computer engineering studies at <a href="https://www.uab.cat/">UAB (Universitat Autònoma de Barcelona)</a>.</p>
                </TimeLineItem>
                <TimeLineItem>
                    <MarkedH3>2017 Oct - 2018 Sep</MarkedH3>
                    <p>I worked for 1 year at <a href="https://eqtic.net/">EQTic</a> developing web and IoT-related applications using technologies such as Java, Spring, Django, Node-Red, NodeJS, and ExpressJS.</p>
                </TimeLineItem>
                <TimeLineItem>
                    <MarkedH3>2016-2018</MarkedH3>
                    <p>I graduated from <a href="https://copernic.cat/">Nicolau Copèrnic</a> with a CFGS in Cross-platform Application Development.</p>
                </TimeLineItem>
                <Separator />
            </ul>
        </div>
    </Layout>
)

export default AboutPage
function TimeLineItem({ children }) {
    return (
        <li>
            <TimeLineItemContent>
                {children}
            </TimeLineItemContent>
        </li>
    )
}

