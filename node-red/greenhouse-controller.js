const fs = require("fs");
const path = require("path");
const { execFile } = require("child_process");

module.exports = function(RED) {
  function GreenhouseControllerNode(config) {
    RED.nodes.createNode(this, config);
    const node = this;

    node.on("input", function(msg, send, done) {
      msg.payload = msg.payload || {};
      if (!msg.payload.controller && config.controllerMode) {
        msg.payload.controller = config.controllerMode;
      }

      const projectPath = config.projectPath || process.cwd();
      let pythonCommand = config.pythonPath || "python3";

      if (!config.pythonPath) {
        const localVenvPython = path.join(projectPath, ".venv", "bin", "python");
        if (fs.existsSync(localVenvPython)) {
          pythonCommand = localVenvPython;
        }
      }

      if (!fs.existsSync(projectPath)) {
        const err = new Error(`Project path does not exist: ${projectPath}`);
        node.error(err.message, msg);
        if (done) done(err);
        return;
      }

      const scriptPath = path.join(projectPath, "src", "predict_controller.py");
      if (!fs.existsSync(scriptPath)) {
        const err = new Error(`Prediction script not found: ${scriptPath}`);
        node.error(err.message, msg);
        if (done) done(err);
        return;
      }

      if (pythonCommand.includes("/") && !fs.existsSync(pythonCommand)) {
        const err = new Error(`Python path does not exist: ${pythonCommand}. In terminal, run: cd ${projectPath} && source .venv/bin/activate && which python`);
        node.error(err.message, msg);
        if (done) done(err);
        return;
      }

      const payload = JSON.stringify(msg.payload);

      execFile(pythonCommand, [scriptPath, payload], { cwd: projectPath }, function(error, stdout, stderr) {
        if (error) {
          node.error(stderr || error.message, msg);
          if (done) done(error);
          return;
        }

        try {
          msg.payload = JSON.parse(stdout);
          send(msg);
          if (done) done();
        } catch (parseError) {
          node.error(stdout || parseError.message, msg);
          if (done) done(parseError);
        }
      });
    });
  }

  RED.nodes.registerType("greenhouse-controller", GreenhouseControllerNode);
};
