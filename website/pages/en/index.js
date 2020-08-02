/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const MarkdownBlock = CompLibrary.MarkdownBlock; /* Used to read markdown */
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

class HomeSplash extends React.Component {
  render() {
    const {siteConfig, language = ''} = this.props;
    const {baseUrl, docsUrl} = siteConfig;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    const docUrl = doc => `${baseUrl}${docsPart}${langPart}${doc}`;

    const SplashContainer = props => (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const Logo = props => (
      <div className="projectLogo">
        <img src={props.img_src} alt=" " />
      </div>
    );

    const ProjectTitle = props => (
      <h2 className="projectTitle">
        {props.title}
        <small>{props.tagline}</small>
      </h2>
    );

    const PromoSection = props => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
        </div>
      </div>
    );

    const Button = props => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
        <Logo img_src={`${baseUrl}img/android-chrome-512x512.png`} />
        <div className="inner">
          <ProjectTitle tagline={siteConfig.tagline} title={siteConfig.title} />
          <PromoSection>
            <Button href="#install">Quick Start</Button>
            <Button href={docUrl('install.html')}>Installation</Button>
            <Button href={docUrl('howto.html')}>How to Doodle</Button>
            <Button href={"https://github.com/dbuscombe-usgs/doodle_labeller"}>Github Repository</Button>
          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}

class Index extends React.Component {
  render() {
    const {config: siteConfig, language = ''} = this.props;
    const {baseUrl} = siteConfig;

    const Block = props => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}>
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );

    // const FeatureCallout = () => (
    //   <div
    //     className="productShowcaseSection paddingBottom"
    //     style={{textAlign: 'center'}}>
    //     <h2>Feature Callout</h2>
    //     <MarkdownBlock>These are features of this project</MarkdownBlock>
    //   </div>
    // );

    const TryOut = () => (
      <Block id="try">
        {[
          {
            content:
              'This tool is also set up to tackle image labeling in stages, using minimal annotations. ' +
              '<br><br>For example, by labeling individual classes then using the resulting binary label images ' +
              'as masks for the imagery to be labeled for subsequent classes. Labeling is achieved using the `doodler.py` script' +
              '<br><br>Label images that are outputted by `doodler.py` can be merged using `merge.py`.',

            image: `${baseUrl}img/im1.png`,
            imageAlign: 'left',
            title: '`doodler.py` (and `merge.py`)',
          },
        ]}
      </Block>
    );

// href={docUrl('doc1.html')}

    const Description = () => (
      <Block id="install" background="light">
        {[
          {
            content:
             'These brief instructions are for regular python and command line users.  <br><br><br>'+
              'Clone the repo:<br>'+
              '`git clone --depth 1 https://github.com/dbuscombe-usgs/doodle_labeller.git`'+
              '<br><br>Make a conda environment:<br>'+
              '`conda env create -f doodler.yml`'+
              '<br><br>Activate the conda environment:<br>'+
              '`conda activate doodler`'+
              '<br><br>Doodle!<br>'+
              '`python doodler.py -c config_file.json`',
            image: `${baseUrl}img/im3.png`,
            imageAlign: 'right',
            title: 'Quick Start',
          },
        ]}
      </Block>
    );


    const LearnHow = () => (
      <Block background="light">
        {[
          {
            content:
              '`Doodler` is a tool for "exemplative", not exhaustive, labeling.'+
              '<br><br>The approach taken here is to freehand label only some of the scene, then use a model to complete the scene. '+
              'Your sparse annotations are used to create an ensemble of Conditional Random Field (CRF) models, '+
              'each of which develops a scene-specific model for each class and creates a dense (i.e. per pixel) label image'+
              'based on the information you provide it. The ensembles are combined for a stable estimate.'+
              '<br><br>This approach can reduce the time required for detailed labeling of large and complex scenes '+
              'by an order of magnitude or more.',
            image: `${baseUrl}img/im6.png`,
            imageAlign: 'right',
            title: 'What doodler does',
          },
        ]}
      </Block>
    );

    const Intro = () => (
      <Block background="light">
        {[
          {
            content:
              'There are many great tools for exhaustive (i.e. whole image) image labeling for segmentation tasks, using polygons. '+
              'Examples include **www.makesense.ai** and **https://cvat.org**.'+
              '<br><br>However, for high-resolution imagery with large spatial footprints and complex scenes, '+
              'such as aerial and satellite imagery, '+
              'exhaustive labeling using polygonal tools can be very time-consuming. '+
              'This is especially true of scenes with many classes of interest, '+
              'and covering relatively small, spatially discontinuous regions of the image.'+
              '<br><br>`Doodler` is for rapid semi-supervised approximate segmentation of such imagery. '+
              'It can reduce the time required for detailed labeling of large and complex scenes '+
              'by an order of magnitude or more.',
            image: `${baseUrl}img/im5.png`,
            imageAlign: 'right',
            title: 'Semi-supervised segmentation of natural scenes',
          },
        ]}
      </Block>
    );

//

    const Features = () => (
      <Block layout="fourColumn">
        {[
          {
            content: 'Freehand label only some of the scene, then use a model to complete the scene. A semi-supervised tool for efficient image labeling, based on sparse examples provided by a human annotator. Those sparse annotations are used by a secondary automated process to estimate the class of every pixel in the image.',
            image: `${baseUrl}img/undraw_pen_nqf7.png`,
            imageAlign: 'top',
            title: 'Label by example',
          },
          {
            content: 'For high-resolution imagery with large spatial footprints and complex scenes, such as aerial and satellite imagery, exhaustive labeling using polygonal tools can be prohibitively time-consuming. Doodler offers a potential alternative.',
            image: `${baseUrl}img/undraw_images_aef7.png`,
            imageAlign: 'top',
            title: 'For rapid approximate image segmentation',
          },
        ]}
      </Block>
    );


    const Acknowledgements = () => (
      <Block layout="fourColumn">
        {[
          {
            // content: 'Freehand label only some of the scene, then use a model to complete the scene. A semi-supervised tool for efficient image labeling, based on sparse examples provided by a human annotator. Those sparse annotations are used by a secondary automated process to estimate the class of every pixel in the image.',
            content:
            '`Doodler` is written and maintained by Daniel Buscombe, Marda Science.'+
            '<br><br> `Doodler` development is funded by the U.S. Geological Survey Coastal Hazards Program, '+
            'and is for the primary usage of U.S. Geological Survey scientists, researchers and affiliated '+
            'colleagues working on the Hurricane Florence Supplemental Project and other coastal hazards research.'+
            '<br><br> Many people have contributed ideas, code fixes, and bug reports. Thanks especially to Jon Warrick, Chris Sherwood, Jenna Brown, Andy Ritchie, Jin-Si Over, Christine Kranenburg, '+
            'and the rest of the Florence Supplemental team; to Evan Goldstein and colleagues at University of North Carolina Greensboro; '+
            'Leslie Hsu at the USGS Community for Data Integration; and LCDR Brodie Wells, formerly of Naval Postgraduate School, Monterey.',

            image: `${baseUrl}img/undraw_team_ih79.svg`,
            imageAlign: 'top',
            title: 'Acknowledgements',
          },
          {
            // content: 'For high-resolution imagery with large spatial footprints and complex scenes, such as aerial and satellite imagery, exhaustive labeling using polygonal tools can be prohibitively time-consuming. Doodler offers a potential alternative.',
            content:
            'The software is optimized for specific types of imagery (see &#8592;), but is highly configurable '+
            'to specific purposes, and is therefore made publicly under an MIT license in the spirit of open source &#10003;, '+
            'open access &#10003;, scientific rigour &#10003; and transparency &#10003;. '+
            '<br><br> While Marda Science cannot carry out unpaid consulting over specific use cases, we encourage you to '+
            'submit issues and new feature requests, and, if you find it useful and have made improvements, to '+
            'contribute to its development through '+
            'a pull request on https://github.com/dbuscombe-usgs/doodle_labeller. ',
            image: `${baseUrl}img/undraw_developer_activity_bv83.svg`,
            imageAlign: 'top',
            title: 'Contributing',
          },
        ]}
      </Block>
    );


    const Warning = () => (
      <Block layout="twoColumn">
        {[
          {
            content:
            '`Doodler` is still in active development and `beta` version. '+
            'Use at your own risk! '+
            '<br><br> Please check back later, `watch` the github repository to receive alerts, '+
            'or listen to announcements on https://twitter.com/magic_walnut for the first official release.',
            image: `${baseUrl}img/undraw_under_construction_46pa.svg`,
            imageAlign: 'top',
            title: 'Warning!',
          },
        ]}
      </Block>
    );

// **Warning**:

    // const Showcase = () => {
    //   if ((siteConfig.users || []).length === 0) {
    //     return null;
    //   }
    //
    //   const showcase = siteConfig.users
    //     .filter(user => user.pinned)
    //     .map(user => (
    //       <a href={user.infoLink} key={user.infoLink}>
    //         <img src={user.image} alt={user.caption} title={user.caption} />
    //       </a>
    //     ));
    //
    //   const pageUrl = page => baseUrl + (language ? `${language}/` : '') + page;
    //
    //   return (
    //     <div className="productShowcaseSection paddingBottom">
    //       <h2>Who is Using This?</h2>
    //       <p>This project is used by all these people</p>
    //       <div className="logos">{showcase}</div>
    //       <div className="more-users">
    //         <a className="button" href={pageUrl('users.html')}>
    //           More {siteConfig.title} Users
    //         </a>
    //       </div>
    //     </div>
    //   );
    // };

    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="mainContainer">
          <Warning />
          <Intro />
          <Features />
          <LearnHow />
          <TryOut />
          <Description />
          <Acknowledgements />
        </div>
      </div>
    );
  }
}

module.exports = Index;
