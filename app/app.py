if __name__ == '__main__':
    from .__init__ import create_app
    app = create_app()
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=5000)