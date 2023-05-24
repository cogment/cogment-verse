# Tunnel using ngrok

[ngrok](https://ngrok.com) is a tool providing a _quick n' dirty_ way to deploy any web service from a host (even your local computer) to the internet. Using ngrok and Cogment Verse, you can easily make an experiment available to the internet. THis will work fine for turn by turn environment but don't expect to have a good level of playability on anything real time as ngrok adds a significant lag.

## Step 1 - Install and setup ngrok

Head to [ngrok](https://ngrok.com) website and signup for a free account. Once this is done, [install ngrok](https://ngrok.com/download) following the official instructions.

Once both of those steps are completed we will need to authenticate the local ngrok instance with your account. To facilitate later configuration, we will create a specific config file at the same time. Run the following, replacing the placeholder with the auth token you can retrieve at <https://dashboard.ngrok.com/get-started/your-authtoken>

```console
$ ngrok --config=./ngrok.yaml config add-authtoken <YOUR_AUTH_TOKEN>
```

This will create a `./ngrok.yaml` file storing this authtoken. Make sure to never share this file.

## Step 2 - Configure the tunnels

To make a Cogment Verse instance accessible through the internet, you'll need to open two tunnels:

- one for the Cogment Verse Web App, that will run locally on port 8080,
- one for the Cogment Orchestrator, that will run locally on port 8081.

Edit the `ngrok.yaml` to add those tunnels configuration.

```yaml
version: "2"
authtoken: <YOUR_AUTH_TOKEN>
tunnels:
  cogment_verse_web_app:
    proto: http
    addr: 8080
  cogment_orchestrator:
    proto: http
    addr: 8081
```

You can now start ngrok for those two tunnels in a dedicated terminal.

```console
$ ngrok --config=./ngrok.yaml start --all
```

This will generate public URLs and start the tunnels. please note two of its output:

- The URL forwarding to `http://localhost:8080` is where the Cogment Verse Web App can be reached, we will call it <COGMENT_VERSE_WEB_APP_ENDPOINT>,
- The URL forwarding to `http://localhost:8081` is where the Cogment Orchestrator can be reached, we will call it <COGMENT_ORCHESTRATOR_ENDPOINT>.

> ⚠️ These URLs will change whenever you restart ngrok.

## Step 3 - Launch a run

You can now launch a Cogment Verse experiment as usual with those additional configuration values:

- `services.web.port=8080`,
- `services.orchestrator.web_port=8081`,
- `services.orchestrator.web_endpoint=<COGMENT_ORCHESTRATOR_ENDPOINT>`.

For example:

```console
$ python -m main \
  +experiment=simple_dqn/connect_four \
  +run.hill_training_trials_ratio=0.1 \
  services.web.port=8080 \
  services.orchestrator.web_port=8081 \
  services.orchestrator.web_endpoint=<COGMENT_ORCHESTRATOR_ENDPOINT>
```

You can then connect to the run for everywhere at `<COGMENT_VERSE_WEB_APP_ENDPOINT>`.
