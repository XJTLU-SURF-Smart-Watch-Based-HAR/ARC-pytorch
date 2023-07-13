import datetime

time = datetime.datetime.now()


def init():
    with open('logs/HAR.log', 'w') as logger:
        logger.write(f'Time: {datetime.datetime.now()}')
        logger.write('\n')


def info(content, end='\n'):
    with open('logs/HAR.log', 'a') as logger:
        print(content)
        logger.write(content)
        logger.write(end)


def end():
    with open('logs/HAR.log', 'a') as logger:
        logger.write(f'Time consumed: {datetime.datetime.now() - time}')
