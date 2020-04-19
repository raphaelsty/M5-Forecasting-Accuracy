from .__init__ import create_app


if __name__ == '__main__':
    app = create_app()
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)